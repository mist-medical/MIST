# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Tests for the Predictor class."""
import torch
from unittest.mock import MagicMock
from mist.inference.predictor import Predictor


class DummyInferer:
    """Dummy inferer that returns the model's output."""
    def __call__(self, image: torch.Tensor, model):
        return model(image)


class DummyEnsembler:
    """Dummy ensembler that averages predictions."""
    def __call__(self, predictions):
        return sum(predictions) / len(predictions)


class DummyTransform:
    """Dummy TTA transform that adds and subtracts 1."""
    def __init__(self):
        self.name = "dummy"

    def __call__(self, image: torch.Tensor):
        return image + 1

    def inverse(self, prediction: torch.Tensor):
        return prediction - 1


def test_predictor_basic_flow():
    """Test basic predictor behavior with one model and one TTA transform."""
    image = torch.ones(1, 1, 4, 4, 4)
    model = lambda x: x * 2

    predictor = Predictor(
        models=[model],
        inferer=DummyInferer(),
        ensembler=DummyEnsembler(),
        tta_transforms=[DummyTransform()],
        device="cpu"
    )

    prediction = predictor(image)
    expected = image + 2  # 1 → +1 → *2 → -1 = 3
    assert torch.allclose(prediction, expected)


def test_predictor_multiple_models_and_tta():
    """Test predictor with two models and two TTA transforms."""
    image = torch.ones(1, 1, 4, 4, 4)

    def model1(x): return x + 2
    def model2(x): return x * 3

    class Identity:
        def __call__(self, x): return x
        def inverse(self, x): return x

    class AddOne:
        def __call__(self, x): return x + 1
        def inverse(self, x): return x - 1

    predictor = Predictor(
        models=[model1, model2],
        inferer=DummyInferer(),
        ensembler=DummyEnsembler(),
        tta_transforms=[Identity(), AddOne()],
        device="cpu"
    )

    prediction = predictor(image)
    expected = torch.full_like(image, 3.5)  # Corrected mean
    assert torch.allclose(prediction, expected)
