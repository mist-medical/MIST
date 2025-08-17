# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Registry for learning rate schedulers used in training."""
from typing import Callable, Dict, List
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# MIST imports.
from mist.training.lr_schedulers.lr_schedulers_constants import LRSchedulerConstants


def _cosine_scheduler(optimizer: Optimizer, epochs: int) -> _LRScheduler:
    """Cosine annealing LR schedule."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def _polynomial_scheduler(optimizer: Optimizer, epochs: int) -> _LRScheduler:
    """Polynomial decay LR schedule."""
    torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=epochs,
        power=LRSchedulerConstants.POLYNOMIAL_DECAY,
    )


def _constant_scheduler(optimizer: Optimizer, epochs: int) -> _LRScheduler:
    """Constant learning rate schedule."""
    return torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=LRSchedulerConstants.CONSTANT_LR_FACTOR
    )


LR_SCHEDULER_REGISTRY: Dict[str, Callable[..., _LRScheduler]] = {
    "cosine": _cosine_scheduler,
    "polynomial": _polynomial_scheduler,
    "constant": _constant_scheduler,
}


def get_lr_scheduler(
    name: str,
    optimizer: Optimizer,
    epochs: int,
) -> _LRScheduler:
    """Factory function for learning rate schedulers."""
    name = name.lower()
    if name not in LR_SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler '{name}'. "
            f"Available: {list_lr_schedulers()}"
        )
    return LR_SCHEDULER_REGISTRY[name](optimizer, epochs)


def list_lr_schedulers() -> List[str]:
    """Return the list of available scheduler names."""
    return sorted(LR_SCHEDULER_REGISTRY.keys())
