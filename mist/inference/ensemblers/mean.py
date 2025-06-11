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
"""Mean-based softmax ensembler for MIST inference."""
from typing import List
import torch

# MIST imports.
from mist.inference.ensemblers.base import AbstractEnsembler
from mist.inference.ensemblers.ensembler_registry import register_ensembler


@register_ensembler("mean")
class MeanEnsembler(AbstractEnsembler):
    """Simple averaging ensembler over softmax outputs."""
    def combine(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Overrides the combine method to average predictions."""
        if not predictions:
            raise ValueError("MeanEnsembler requires at least one prediction.")

        stacked = torch.stack(predictions, dim=0)     # Shape: (N, C, D, H, W)
        mean_prediction = torch.mean(stacked, dim=0)  # Shape: (C, D, H, W)
        return mean_prediction

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
