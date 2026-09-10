import copy
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from objectives.losses.caption_alignment import CaptionAlignmentObjective
from objectives.losses.caption_sigmoid import CaptionSigmoidObjective


class TextEncoder(nn.Module):
    def forward(self, captions):
        table = torch.tensor([[1., .2, -.1], [.8, .1, .3],
                              [-.2, 1., .1], [.1, .7, .5]])
        return table[[int(c) for c in captions]]


def make_objective(mode="pid", reduction="anchor", gather=False):
    return CaptionSigmoidObjective(
        image_dim=3, text_dim=3, text_encoder=TextEncoder(),
        use_projection=False, temperature=.5, positive_mode=mode,
        sigmoid_bias=-.4, sigmoid_reduction=reduction,
        gather_across_ranks=gather,
    )


class Graph(nn.Module):
    def __init__(self, mode, reduction, gather):
        super().__init__()
        self.project = nn.Linear(3, 3, bias=False)
        self.objective = make_objective(mode, reduction, gather)

    def forward(self, x, captions, mask, pids):
        return self.objective(self.project(x), captions, mask, pids)


def ddp_worker(rank, url):
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=url, timeout=timedelta(seconds=45))
    try:
        x = torch.tensor([[1., .3, .2], [.7, -.1, .3],
                          [.2, .9, -.1], [-.3, .5, .7]])
        pids = torch.tensor([0, 0, 1, 1])
        for mode in ("instance", "pid"):
            for reduction in ("anchor", "pair_mean", "balanced"):
                for empty_rank in (False, True):
                    torch.manual_seed(123)
                    reference = Graph(mode, reduction, False)
                    distributed = copy.deepcopy(reference)
                    distributed.objective.gather_across_ranks = True
                    wrapper = nn.parallel.DistributedDataParallel(
                        distributed, broadcast_buffers=False)
                    mask = torch.tensor([not empty_rank, True, True, True])
                    # Unequal local sizes; first rank has no valid caption
                    # in the empty_rank case.
                    sl = slice(0, 1) if rank == 0 else slice(1, 4)
                    captions = [str(i) for i in range(4)]
                    expected = reference(x, captions, mask, pids)
                    expected.backward()
                    actual = wrapper(x[sl], captions[sl], mask[sl], pids[sl])
                    actual.backward()
                    report = actual.detach().clone()
                    dist.all_reduce(report)
                    torch.testing.assert_close(report / 2, expected.detach(),
                                               rtol=2e-5, atol=2e-6)
                    for (_, p), (_, q) in zip(distributed.named_parameters(),
                                              reference.named_parameters()):
                        torch.testing.assert_close(p.grad, q.grad,
                                                   rtol=3e-5, atol=3e-6)
    finally:
        dist.destroy_process_group()


class SigmoidTest(unittest.TestCase):
    def test_reductions_and_label_modes(self):
        x = torch.tensor([[1., .3, .2], [.7, -.1, .3],
                          [.2, .9, -.1], [-.3, .5, .7]], requires_grad=True)
        pids = torch.tensor([0, 0, 1, 1])
        for mode in ("pid", "instance"):
            for reduction in ("anchor", "pair_mean", "balanced"):
                obj = make_objective(mode, reduction)
                logits = obj._compute_logits(F.normalize(x, dim=-1),
                    F.normalize(obj.text_encoder(["0", "1", "2", "3"]), dim=-1))
                mask = pids[:, None].eq(pids[None, :]) if mode == "pid" else torch.eye(4).bool()
                pair = F.binary_cross_entropy_with_logits(logits, mask.float(), reduction="none")
                if reduction == "anchor":
                    expected = pair.sum() / 4
                elif reduction == "pair_mean":
                    expected = pair.mean()
                else:
                    expected = .5 * (pair[mask].mean() + pair[~mask].mean())
                actual = obj(x, ["0", "1", "2", "3"], torch.ones(4).bool(), pids)
                torch.testing.assert_close(actual, expected)
                actual.backward(retain_graph=True)
                self.assertTrue(torch.isfinite(obj.logit_bias.grad))

    def test_extreme_logits_and_missing_sign(self):
        obj = make_objective(reduction="balanced")
        logits = torch.tensor([[1000., -1000.]], requires_grad=True)
        for mask in (torch.ones_like(logits).bool(), torch.zeros_like(logits).bool()):
            loss = obj._alignment_loss(logits, mask)
            expected = F.softplus(-logits if mask.all() else logits).mean()
            torch.testing.assert_close(loss, expected)
            self.assertTrue(torch.isfinite(loss))
        self.assertEqual(obj._alignment_loss(logits[:0], torch.zeros_like(logits[:0]).bool()).item(), 0)

    def test_nce_matches_original_formula(self):
        obj = CaptionAlignmentObjective(3, 3, TextEncoder(), use_projection=False)
        logits = torch.randn(4, 4)
        mask = torch.tensor([0, 0, 1, 1])[:, None].eq(torch.tensor([0, 0, 1, 1])[None, :])
        expected = -(logits.softmax(1) * mask).sum(1).log().mean()
        torch.testing.assert_close(obj._alignment_loss(logits, mask), expected)
        torch.testing.assert_close(obj._alignment_loss(logits, torch.eye(4).bool()),
                                   F.cross_entropy(logits, torch.arange(4)))

    def test_checkpoint_and_fixed_parameters(self):
        obj = make_objective()
        restored = make_objective()
        restored.load_state_dict(obj.state_dict())
        self.assertIn("logit_scale", obj.state_dict())
        self.assertIn("logit_bias", obj.state_dict())
        fixed = CaptionSigmoidObjective(3, 3, TextEncoder(),
                    use_projection=False, sigmoid_learnable=False)
        self.assertFalse(list(fixed.parameters()))

    def test_all_invalid_is_differentiable_zero(self):
        obj = make_objective()
        images = torch.randn(2, 3, requires_grad=True)
        value = obj(images, ["0", "1"], torch.zeros(2).bool(), torch.tensor([0, 1]))
        self.assertEqual(value.item(), 0)
        value.backward()
        self.assertEqual(obj.logit_scale.grad.item(), 0)
        self.assertEqual(obj.logit_bias.grad.item(), 0)
        self.assertTrue(torch.equal(images.grad, torch.zeros_like(images)))

    def test_invalid_reduction(self):
        with self.assertRaises(ValueError):
            make_objective(reduction="unknown")

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
    def test_two_rank_values_and_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            url = (Path(tmp) / "rendezvous").as_uri()
            mp.spawn(ddp_worker, args=(url,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
