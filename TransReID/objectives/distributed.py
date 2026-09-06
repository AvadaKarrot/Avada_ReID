"""Small distributed primitives used by global-batch objectives.

The helpers deliberately gather candidates while leaving anchors rank-local.
Consequently each rank materializes ``local_batch x global_batch`` pairwise
scores instead of redundantly materializing the full global square matrix.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist


def distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


class _DifferentiableAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()
        outputs = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(outputs, tensor.contiguous())
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Input from rank r is consumed as output r on every rank. Sum those
        # contributions before returning the gradient to its owner.
        stacked = torch.stack(
            [gradient.contiguous() for gradient in grad_outputs], dim=0
        )
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
        return stacked[ctx.rank]


@dataclass(frozen=True)
class GatherLayout:
    sizes: Tuple[int, ...]
    local_offset: int
    global_size: int


def _gather_sizes(local_size: int, device: torch.device) -> GatherLayout:
    if not distributed_ready():
        return GatherLayout((local_size,), 0, local_size)
    size = torch.tensor([local_size], device=device, dtype=torch.long)
    gathered = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, size)
    sizes = tuple(int(item.item()) for item in gathered)
    rank = dist.get_rank()
    return GatherLayout(sizes, sum(sizes[:rank]), sum(sizes))


def gather_variable_with_grad(tensor: torch.Tensor):
    """Differentiably gather a variable-length leading dimension."""
    layout = _gather_sizes(tensor.shape[0], tensor.device)
    if not distributed_ready():
        return tensor, layout
    max_size = max(layout.sizes, default=0)
    if max_size == 0:
        return tensor, layout
    if tensor.shape[0] < max_size:
        padding = tensor.new_zeros(
            (max_size - tensor.shape[0],) + tensor.shape[1:]
        )
        tensor = torch.cat((tensor, padding), dim=0)
    gathered = _DifferentiableAllGather.apply(tensor)
    return torch.cat(
        [part[:size] for part, size in zip(gathered, layout.sizes)], dim=0
    ), layout


@torch.no_grad()
def gather_variable(tensor: torch.Tensor, layout: GatherLayout = None):
    """Gather labels or metadata without adding an autograd edge."""
    actual_layout = layout or _gather_sizes(tensor.shape[0], tensor.device)
    if not distributed_ready():
        return tensor, actual_layout
    max_size = max(actual_layout.sizes, default=0)
    if max_size == 0:
        return tensor, actual_layout
    if tensor.shape[0] < max_size:
        padding = tensor.new_zeros(
            (max_size - tensor.shape[0],) + tensor.shape[1:]
        )
        tensor = torch.cat((tensor, padding), dim=0)
    gathered = [torch.empty_like(tensor) for _ in actual_layout.sizes]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(
        [part[:size] for part, size in zip(gathered, actual_layout.sizes)],
        dim=0,
    ), actual_layout


def globally_normalized_local_sum(
    local_values: torch.Tensor,
) -> torch.Tensor:
    """Return a scalar whose DDP-averaged gradient is a global mean."""
    if not distributed_ready():
        if local_values.numel() == 0:
            return local_values.sum() * 0.0
        return local_values.mean()
    count = torch.tensor(
        [local_values.numel()], device=local_values.device, dtype=torch.long
    )
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    global_count = int(count.item())
    if global_count == 0:
        return local_values.sum() * 0.0
    return (
        local_values.sum()
        * dist.get_world_size()
        / float(global_count)
    )
