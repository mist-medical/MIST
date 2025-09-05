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
"""Alpha schedulers for dynamic composite loss weighting."""


class ConstantScheduler:
    """Constant alpha scheduler.

    This scheduler maintains a constant alpha value throughout training.

    Attributes:
        value: The constant alpha value to be used.
    """
    def __init__(self, num_epochs: int,  value: float=0.5):
        """Initialize the ConstantScheduler.

        Args:
            value: The constant alpha value to be used.
        """
        self.value = float(value)

    def __call__(self, epoch: int) -> float:
        """Get the current alpha value.

        Args:
            epoch: Current epoch number (ignored).

        Returns:
            The constant alpha value.
        """
        return self.value


class LinearScheduler:
    """Linear schedule with optional warmup.

    This scheduler sets alpha = 1 for `init_pause` epochs, then linearly decays
    to 0.

    Attributes:
        total_epochs: Total number of epochs for training.
        init_pause: Number of initial epochs to keep alpha at 1.
        decay_epochs: Number of epochs over which to linearly decay alpha to 0.
    """
    def __init__(self, num_epochs: int, init_pause: int=5):
        """Initialize the LinearScheduler.

        Args:
            num_epochs: Total number of epochs for training.
            init_pause: Number of initial epochs to keep alpha at 1.
        """
        self.total_epochs = max(1, num_epochs - 1)
        self.init_pause = init_pause
        self.decay_epochs = max(1, self.total_epochs - self.init_pause)

    def __call__(self, epoch: int) -> float:
        """Get the current alpha value.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            The alpha value for the current epoch.
        """
        if epoch <= self.init_pause:
            return 1.0
        progress = (epoch - self.init_pause) / self.decay_epochs
        return max(0.0, 1.0 - progress)


ALPHA_SCHEDULER_REGISTRY = {
    "constant": ConstantScheduler,
    "linear": LinearScheduler,
}


def get_alpha_scheduler(name: str, **kwargs):
    """Factory to get alpha scheduler by name.

    Args:
        name: Name of the scheduler.
        **kwargs: Keyword arguments to pass to the scheduler.

    Returns:
        An instance of the requested scheduler.

    Raises:
        ValueError: If the scheduler name is unrecognized.
    """
    try:
        return ALPHA_SCHEDULER_REGISTRY[name](**kwargs)
    except KeyError as e:
        raise ValueError(f"Unknown scheduler: {name}") from e


def list_alpha_schedulers():
    """List available alpha scheduler names."""
    return list(ALPHA_SCHEDULER_REGISTRY.keys())
