"""Image-only target-domain evaluation entry point."""

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import cfg
from data.build import build_datamanager
from engine import Evaluator
from evaluation import write_results
from modeling import build_model


def main():
    parser = argparse.ArgumentParser(description="Unified ReID evaluation")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    data_manager = build_datamanager(cfg)
    model = build_model(cfg, num_classes=data_manager._num_train_pids)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.to(cfg.MODEL.DEVICE)

    evaluator = Evaluator(
        device=cfg.MODEL.DEVICE,
        feat_norm=cfg.TEST.FEAT_NORM == "yes",
    )
    results = evaluator.evaluate_loader(
        model,
        data_manager.test_loader,
        data_manager.num_query,
    )
    if args.output:
        write_results(results, args.output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
