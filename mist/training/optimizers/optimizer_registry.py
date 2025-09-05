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
"""Registry for optimizers used in training."""
from typing import Callable, Dict, List
from collections import OrderedDict
import torch
from torch.optim import Optimizer

# MIST imports.
from mist.training.optimizers.optimizer_constants import OptimizerConstants


def _adam_optimizer(
    params: OrderedDict,
    learning_rate: float,
    weight_decay: float,
    eps: float,
) -> Optimizer:
    """Internal Adam optimizer."""
    return torch.optim.Adam(
        params=params, lr=learning_rate, weight_decay=weight_decay, eps=eps
    )


def _adamw_optimizer(
    params: OrderedDict,
    learning_rate: float,
    weight_decay: float,
    eps:float,
) -> Optimizer:
    """Internal AdamW optimizer."""
    return torch.optim.AdamW(
        params=params, lr=learning_rate, weight_decay=weight_decay, eps=eps
    )


def _sgd_optimizer(
    params: OrderedDict,
    learning_rate: float,
    weight_decay: float,
    eps: float, # pylint:disable=unused-argument
) -> Optimizer:
    """SGD optimizer."""
    return torch.optim.SGD(
        params=params,
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=OptimizerConstants.SGD_MOMENTUM,
        nesterov=OptimizerConstants.SGD_NESTEROV
    )


OPTIMIZER_REGISTRY: Dict[str, Callable[..., Optimizer]] = {
    "adam": _adam_optimizer,
    "adamw": _adamw_optimizer,
    "sgd": _sgd_optimizer,
}


def get_optimizer(name: str, params: OrderedDict, **kwargs) -> Optimizer:
    """Factory function for optimizers.

    Args:
        name: Optimizer name (adam, adamw, sgd).
        params: Model parameters to optimize.
        **kwargs: Optimizer-specific keyword arguments.

    Returns:
        torch.optim.Optimizer

    Raises:
        ValueError: If optimizer name is invalid.
    """
    name = name.lower()
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Available: {list_optimizers()}"
        )
    return OPTIMIZER_REGISTRY[name](params, **kwargs)


def list_optimizers() -> List[str]:
    """Return the list of available optimizer names."""
    return sorted(OPTIMIZER_REGISTRY.keys())
