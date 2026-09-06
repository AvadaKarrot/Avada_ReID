"""Two-rank CPU smoke test for global Caption and Triplet candidate pools."""

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from objectives.losses.batch_hard_triplet import BatchHardTripletLoss
from objectives.losses.caption_alignment import CaptionAlignmentObjective


class _TextEncoder(nn.Module):
    output_dim = 2

    def forward(self, captions):
        rows = []
        for caption in captions:
            rows.append((1.0, 0.0) if caption == "left" else (0.0, 1.0))
        return torch.tensor(rows, dtype=torch.float32)


def _worker(rank, world_size, init_method=None):
    if init_method is None:
        dist.init_process_group(backend="gloo")
    else:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
    if dist.get_world_size() != world_size:
        raise RuntimeError(
            f"This smoke test requires exactly {world_size} ranks"
        )

    pids = torch.tensor([rank, rank], dtype=torch.long)
    captions = ("left", "left") if rank == 0 else ("right", "right")
    base = (
        torch.tensor([[1.0, 0.0], [0.9, 0.1]])
        if rank == 0
        else torch.tensor([[0.0, 1.0], [0.1, 0.9]])
    )

    caption_features = base.clone().requires_grad_(True)
    caption = CaptionAlignmentObjective(
        image_dim=2,
        text_dim=2,
        text_encoder=_TextEncoder(),
        use_projection=False,
        positive_mode="pid",
        gather_across_ranks=True,
    )
    caption_loss = caption(
        image_features=caption_features,
        captions=captions,
        valid_mask=torch.ones(2, dtype=torch.bool),
        pids=pids,
    )
    caption_loss.backward()

    triplet_features = base.clone().requires_grad_(True)
    triplet = BatchHardTripletLoss(
        margin=0.3,
        gather_across_ranks=True,
    )
    triplet_loss = triplet(triplet_features, pids)
    triplet_loss.backward()

    for name, value, gradient in (
        ("caption", caption_loss, caption_features.grad),
        ("triplet", triplet_loss, triplet_features.grad),
    ):
        if not torch.isfinite(value):
            raise RuntimeError(f"{name} loss is not finite")
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"{name} gradient is missing or invalid")

    dist.barrier()
    if rank == 0:
        print(
            "GLOBAL_OBJECTIVE_SMOKE_OK "
            f"world_size={dist.get_world_size()} local_batch=2 "
            "global_candidates=4"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if "RANK" in os.environ:
        _worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
    else:
        rendezvous = Path(tempfile.gettempdir()) / "reid-ddp-smoke-rendezvous"
        rendezvous.unlink(missing_ok=True)
        init_method = "file:///" + rendezvous.as_posix().lstrip("/")
        mp.spawn(_worker, args=(2, init_method), nprocs=2, join=True)
