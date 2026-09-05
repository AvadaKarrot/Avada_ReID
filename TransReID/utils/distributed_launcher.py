"""Small, testable helpers for launching unified training with torchrun."""


def wrap_torchrun(command, *, nproc_per_node):
    """Wrap ``python entry.py ...`` in the same interpreter's torchrun."""

    processes = int(nproc_per_node)
    if processes < 1:
        raise ValueError("nproc_per_node must be at least 1")
    if processes == 1:
        return list(command)
    if len(command) < 2:
        raise ValueError("Expected a Python command followed by an entry point")
    return [
        command[0],
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={processes}",
        *command[1:],
    ]


def distributed_overrides(*, nproc_per_node, global_batch_size):
    processes = int(nproc_per_node)
    global_batch = int(global_batch_size)
    if global_batch <= 0:
        raise ValueError("global_batch_size must be positive")
    if global_batch % processes:
        raise ValueError("global batch size must be divisible by process count")
    return [
        "MODEL.DIST_TRAIN",
        "True" if processes > 1 else "False",
        "SOLVER.IMS_PER_BATCH",
        str(global_batch),
    ]


def validate_gpu_ids(gpu_ids, *, nproc_per_node):
    """Require an explicit GPU list to match the requested process count."""

    if not gpu_ids:
        return
    visible = [item.strip() for item in gpu_ids.split(",") if item.strip()]
    if len(visible) != int(nproc_per_node):
        raise ValueError(
            f"gpu_ids exposes {len(visible)} GPUs but nproc_per_node is "
            f"{int(nproc_per_node)}"
        )
