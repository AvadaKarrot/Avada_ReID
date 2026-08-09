from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from ..outputs import BackboneOutput


class BackboneAdapter(nn.Module, ABC):
    """Common interface implemented by every visual backbone."""

    output_dim: int

    @abstractmethod
    def forward_features(self, images: Any) -> BackboneOutput:
        raise NotImplementedError

    def forward(self, images: Any) -> BackboneOutput:
        return self.forward_features(images)
