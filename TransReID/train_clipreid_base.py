import argparse
import os
import random

import numpy as np
import torch

from config import cfg
from data.build import build_datamanager
from loss.make_loss_clipreid import make_loss
from model.make_model_clipreid_base import make_model
from processor.processor_clipreid_base import do_train_clipreid_base
from solver.lr_scheduler import WarmupMultiStepLR
from solver.make_optimizer_clipreid import make_optimizer
from utils.logger import setup_logger


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config-file",
        "--config_file",
        dest="config_file",
        default="configs/experiments/clip_market_to_msmt17.yml",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    set_seed(cfg.SOLVER.SEED)

    local_rank = int(os.getenv("LOCAL_RANK", args.local_rank))
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    if args.config_file:
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r", encoding="utf-8") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    logger.info(
        "Source=%s Target=%s Caption=%s",
        cfg.DATASETS.SOURCES,
        cfg.DATASETS.TARGETS,
        cfg.OBJECTIVE.CAPTION.ENABLED,
    )

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    data_manager = build_datamanager(cfg)

    model = make_model(
        cfg,
        num_class=data_manager._num_train_pids,
        camera_num=data_manager._num_train_cams,
        view_num=0,
    )
    loss_func, center_criterion = make_loss(
        cfg, num_classes=data_manager._num_train_pids
    )
    optimizer, optimizer_center = make_optimizer(
        cfg, model, center_criterion
    )
    scheduler = WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        cfg.SOLVER.WARMUP_FACTOR,
        cfg.SOLVER.WARMUP_ITERS,
        cfg.SOLVER.WARMUP_METHOD,
    )

    do_train_clipreid_base(
        cfg,
        model,
        center_criterion,
        data_manager,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        local_rank,
    )


if __name__ == "__main__":
    main()
