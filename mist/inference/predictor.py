# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Predictor class to chain together inference, TTA, and ensembling."""
from typing import Callable, List, Optional, Union
import torch

# MIST imports.
from mist.inference.inferers.base import AbstractInferer
from mist.inference.ensemblers.base import AbstractEnsembler
from mist.inference.tta.transforms import AbstractTransform


class Predictor:
    """Performs inference with test time augmentation and ensembling."""
    def __init__(
        self,
        models: List[Callable[[torch.Tensor], torch.Tensor]],
        inferer: AbstractInferer,
        ensembler: AbstractEnsembler,
        tta_transforms: List[AbstractTransform],
        device: Optional[Union[str, torch.device]]=None,
    ):
        """Initialize the predictor.

        Args:
            models: List of PyTorch models.
            inferer: A subclass of AbstractInferer.
            ensembler: A subclass of AbstractEnsembler to aggregate outputs.
            tta_transforms: List of TTA transforms. Predefined lists of
                transforms are available as tta strategies in the
                mist.inference.tta.strategies module.
            device: Torch device to use for inference.
        """
        self.models = models
        self.inferer = inferer
        self.ensembler = ensembler
        self.tta_transforms = tta_transforms
        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Call the predictor like a function."""
        return self.predict(image)

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Run prediction with TTA and ensembling.

        Args:
            image: Input tensor of shape (1, C, D, H, W).

        Returns:
            Combined prediction tensor of shape (1, C, D, H, W).
        """
        image = image.to(self.device)
        all_predictions = []

        for model in self.models:
            model_predictions = []

            # Apply TTA if defined.
            for transform in self.tta_transforms:
                augmented_image = transform(image)
                prediction = self.inferer(augmented_image, model)
                restored = transform.inverse(prediction)
                model_predictions.append(restored)

            # Ensembling across TTA variants for this model.
            combined_prediction = self.ensembler(model_predictions)
            all_predictions.append(combined_prediction)

        # Final ensemble across all models.
        return self.ensembler(all_predictions)
