from collections import OrderedDict
from typing import Mapping

import torch

from .batch import move_to_device, normalize_batch

try:
    from utils.metrics import R1_mAP_eval
except ImportError:
    from TransReID.utils.metrics import R1_mAP_eval


class Evaluator:
    """Image-only evaluator for one or many held-out target domains."""

    def __init__(self, device="cuda", feat_norm=True, max_rank=50):
        self.device = torch.device(device)
        self.feat_norm = feat_norm
        self.max_rank = max_rank

    @torch.no_grad()
    def evaluate_loader(self, model, loader, num_query):
        metric = R1_mAP_eval(
            num_query,
            max_rank=self.max_rank,
            feat_norm="yes" if self.feat_norm else "no",
        )
        metric.reset()
        model.eval()

        for raw_batch in loader:
            batch = normalize_batch(raw_batch)
            images = move_to_device(batch["images"], self.device)
            outputs = model(images)
            metric.update(
                (
                    outputs.embedding,
                    batch["pids"],
                    batch["camids"],
                )
            )

        cmc, mAP, *_ = metric.compute()
        return {
            "mAP": float(mAP),
            "rank1": float(cmc[0]),
            "rank5": float(cmc[4]) if len(cmc) >= 5 else None,
            "rank10": float(cmc[9]) if len(cmc) >= 10 else None,
        }

    def evaluate_targets(self, model, targets: Mapping):
        results = OrderedDict()
        for name, target in targets.items():
            results[name] = self.evaluate_loader(
                model,
                loader=target["loader"],
                num_query=target["num_query"],
            )
        return results
