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
"""Abstract base class for all inferers in MIST."""
from abc import ABC, abstractmethod
from typing import Callable, Any
import torch


class AbstractInferer(ABC):
    """Abstract base class for MIST inference methods."""
    def __init__(self):
        self.name = self.__class__.__name__.lower()

    def __call__(
        self,
        image: torch.Tensor,
        model: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Call the inferer like a function."""
        return self.infer(image, model)

    @abstractmethod
    def infer(
        self,
        image: torch.Tensor,
        model: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Perform model inference on a single image."""
        pass # pylint:disable=unnecessary-pass # pragma: no cover

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AbstractInferer) and self.name == other.name
