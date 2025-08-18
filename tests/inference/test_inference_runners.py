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
"""Tests for mist.inference.inference_runners."""
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from pathlib import Path
import copy
import os
import pytest
import torch
import numpy as np
import pandas as pd

# MIST imports.
from mist.inference import inference_runners as ir


@pytest.fixture()
def mock_mist_config():
    """Fixture to provide a mock MIST configuration."""
    return {
        "dataset_info": {
            "modality": "ct",
            "labels": [0, 1],
        },
        "preprocessing": {
            "skip": False,
            "target_spacing": [1.0, 1.0, 1.0],
            "crop_to_foreground": False,
            "normalize_with_nonzero_mask": False,
            "ct_normalization": {
                "window_min": -100.0,
                "window_max": 100.0,
                "z_score_mean": 0.0,
                "z_score_std": 1.0,
            },
            "compute_dtms": False,
            "normalize_dtms": True,
        },
        "model": {
            "architecture": "nnunet",
            "params": {
                "in_channels": 1,
                "out_channels": 2,
                "patch_size": [64, 64, 64],
                "target_spacing": [1.0, 1.0, 1.0],
                "use_deep_supervision": False,
                "use_residual_blocks": False,
                "use_pocket_model": False,
            }
        },
        "training": {
            "seed": 42,
            "hardware": {
                "num_gpus": 2,
                "num_cpu_workers": 8
            }
        },
        "inference": {
            "inferer": {
                "name": "sliding_window",
                "params": {
                    "patch_size": [64, 64, 64],
                    "patch_blend_mode": "gaussian",
                    "patch_overlap": 0.5
                }
            },
            "ensemble": {
                "strategy": "mean"
            },
            "tta": {
                "enabled": True,
                "strategy": "all_flips"
            },
        },
    }


@pytest.fixture
def noop_cuda_tensor_to(monkeypatch):
    """Make Tensor.to('cuda' | torch.device('cuda')) a no-op.
 
    This is so tests can exercise the 'cuda' code path on machines without CUDA
    builds.
    """
    orig_to = torch.Tensor.to

    def _safe_to(self, *args, **kwargs):
        # Support both positional and keyword 'device' args
        device_arg = args[0] if args else kwargs.get("device", None)

        # Normalize to a device type string
        dev_type = None
        if isinstance(device_arg, str):
            dev_type = device_arg
        elif isinstance(device_arg, torch.device):
            dev_type = device_arg.type

        # If asking for CUDA, short-circuit and keep tensor on CPU
        if dev_type == "cuda":
            return self

        # Otherwise, delegate to the real implementation
        return orig_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", _safe_to, raising=True)

class _DummyANTsImage:
    """Minimal stand-in for ANTsImage."""
    def __init__(self, array: np.ndarray | None = None):
        self._array = (
            np.array(0, dtype=np.uint8) if array is None else np.asarray(array)
        )
        self.astype_arg = None
        self.new_like_last_data = None

    def numpy(self):
        """Return the underlying numpy array."""
        return np.asarray(self._array)

    def astype(self, dtype: str):
        """Simulate ANTsImage astype method."""
        # Record and return self to simulate ANTs API style.
        self.astype_arg = dtype
        if dtype == "uint8":
            self._array = self._array.astype(np.uint8)
        return self

    def new_image_like(self, data):
        """Simulate ANTsImage new_image_like method."""
        # Simulate ANTsImage constructor from data.
        self.new_like_last_data = np.asarray(data)
        return _DummyANTsImage(self.new_like_last_data)


def _predictor_logits_two_class(_: torch.Tensor) -> torch.Tensor:
    """Return logits so argmax along channel (axis=1) -> all ones."""
    # Shape: (1, C=2, D=2, H=2, W=2).
    zeros = torch.zeros(1, 1, 2, 2, 2)
    ones = torch.ones(1, 1, 2, 2, 2)
    return torch.cat([zeros, ones], dim=1)


@patch(
    "mist.inference.inference_runners.inference_utils.back_to_original_space"
)
def test_predict_single_example_no_remap_no_crop(
    mock_back_to_original_space,
    mock_mist_config,
    monkeypatch,
):
    """No remap when labels match and no crop; returns uint8 ANTs-like image."""
    # Make config compatible with the function (expects model['out_channels']).
    cfg = copy.deepcopy(mock_mist_config)
    cfg["model"]["params"]["out_channels"] = 2
    cfg["preprocessing"]["crop_to_foreground"] = False
    # Labels match training labels [0, 1] -> no remap path.
    cfg["dataset_info"]["labels"] = [0, 1]

    # Ensure argmax/squeeze axes are known.
    monkeypatch.setattr(
        ir, "ic", SimpleNamespace(ARGMAX_AXIS=1, BATCH_AXIS=0), raising=False
    )

    # Set up back_to_original_space to return an ANTs-like image with astype().
    mocked_ants_img = _DummyANTsImage(np.ones((2, 2, 2), dtype=np.int64))
    mock_back_to_original_space.return_value = mocked_ants_img

    pre_img = torch.randn(1, 1, 2, 2, 2)
    orig_ants = _DummyANTsImage(np.zeros((2, 2, 2), dtype=np.int64))

    out = ir.predict_single_example(
        preprocessed_image=pre_img,
        original_ants_image=orig_ants, # type: ignore
        mist_configuration=cfg,
        predictor=_predictor_logits_two_class, # type: ignore
        foreground_bounding_box=None,
    )

    # back_to_original_space should receive a numpy label map of shape (2, 2, 2)
    # with all ones.
    call_kwargs = mock_back_to_original_space.call_args.kwargs
    assert call_kwargs["original_ants_image"] is orig_ants
    np.testing.assert_array_equal(
        call_kwargs["raw_prediction"], np.ones((2, 2, 2))
    )
    assert call_kwargs["training_labels"] == [0, 1]
    assert call_kwargs["foreground_bounding_box"] is None

    # Returned ANTs-like image should be cast to uint8.
    assert isinstance(out, _DummyANTsImage)
    assert out.astype_arg == "uint8"
    assert out.numpy().dtype == np.uint8


@patch("mist.inference.inference_runners.inference_utils.remap_mask_labels")
@patch(
    "mist.inference.inference_runners.inference_utils.back_to_original_space"
)
@patch("mist.inference.inference_runners.utils.get_fg_mask_bbox")
def test_predict_single_example_with_crop_and_remap(
    mock_get_fg_bbox,
    mock_back_to_original_space,
    mock_remap_labels,
    mock_mist_config,
    monkeypatch,
):
    """Crop bbox computed and labels remapped when original labels differ."""
    # Config: different original labels to trigger remap.
    cfg = copy.deepcopy(mock_mist_config)
    cfg["model"]["out_channels"] = 2
    cfg["preprocessing"]["crop_to_foreground"] = True
    cfg["dataset_info"]["labels"] = [0, 2]  # Different from training [0, 1].

    # Axes constants.
    monkeypatch.setattr(
        ir, "ic", SimpleNamespace(ARGMAX_AXIS=1, BATCH_AXIS=0), raising=False
    )

    # Foreground bbox that should be auto-computed.
    bbox = {"x0": 0, "x1": 1, "y0": 0, "y1": 1, "z0": 0, "z1": 1}
    mock_get_fg_bbox.return_value = bbox

    # back_to_original_space returns an ANTs-like image whose numpy() feeds
    # remapping.
    ants_after_restore = _DummyANTsImage(np.full((2, 2, 2), 1, dtype=np.int64))
    mock_back_to_original_space.return_value = ants_after_restore

    # remap returns a different numpy array; function must wrap it via
    # new_image_like().
    remapped = np.full((2, 2, 2), 2, dtype=np.int64)
    mock_remap_labels.return_value = remapped

    pre_img = torch.randn(1, 1, 2, 2, 2)
    orig_ants = _DummyANTsImage(np.zeros((2, 2, 2), dtype=np.int64))

    out = ir.predict_single_example(
        preprocessed_image=pre_img,
        original_ants_image=orig_ants, # type: ignore
        mist_configuration=cfg,
        predictor=_predictor_logits_two_class, # type: ignore
        foreground_bounding_box=None,  # Force function to compute bbox.
    )

    # get_fg_mask_bbox was used due to crop_to_foreground=True and missing bbox.
    mock_get_fg_bbox.assert_called_once_with(orig_ants)

    # back_to_original_space received the computed bbox and training labels.
    call_kwargs = mock_back_to_original_space.call_args.kwargs
    assert call_kwargs["foreground_bounding_box"] == bbox
    assert call_kwargs["training_labels"] == [0, 1]

    # remap called with restored ANTs image as numpy array and original labels.
    mock_remap_labels.assert_called_once()
    np.testing.assert_array_equal(
        mock_remap_labels.call_args.args[0], ants_after_restore.numpy()
    )
    assert mock_remap_labels.call_args.args[1] == [0, 2]

    # new_image_like must have been used to wrap the remapped numpy back to an
    # image.
    assert isinstance(out, _DummyANTsImage)
    assert orig_ants.new_like_last_data is not None
    np.testing.assert_array_equal(orig_ants.new_like_last_data, remapped)
    assert out.astype_arg == "uint8"


# Tests for test_on_fold.
class _PB:
    """Minimal progress bar stub with context and .track()."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def track(self, it):
        return it


class _DummyModel:
    """Minimal model stub with eval() and to() methods."""
    def __init__(self):
        self.device = None

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self


def _make_train_df(fold: int, image_key: str="image"):
    """Create a minimal DataFrame with patient id, fold, and image path."""
    return pd.DataFrame(
        [{"id": "p1", "fold": fold, image_key: "/tmp/p1.nii.gz"}]
    )


def _make_bbox_df():
    """Create a minimal DataFrame with foreground bounding box for a patient."""
    return pd.DataFrame(
        [{"id": "p1", "x0": 0, "x1": 1, "y0": 0, "y1": 1, "z0": 0, "z1": 1}]
    )


class _DummyLoader:
    """Minimal DALI-like loader with .next() returning [{'image': tensor}]."""
    def __init__(self, n: int):
        self.n = n
        self.i = 0
        self.batch = [{"image": torch.zeros(1, 1, 2, 2, 2)}]

    def next(self):
        # The runner calls exactly len(test_df) times; always return batch.
        if self.i < self.n:
            self.i += 1
        return self.batch


def _prep_dirs(tmp_path: Path):
    """Prepare filesystem layout for test_on_fold test."""
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    (results_dir / "models").mkdir(parents=True, exist_ok=True)
    (results_dir / "predictions" / "train" / "raw").mkdir(
        parents=True, exist_ok=True
    )
    (numpy_dir / "images").mkdir(parents=True, exist_ok=True)
    return results_dir, numpy_dir


@patch(
    "mist.inference.inference_runners.utils.get_progress_bar",
    return_value=_PB()
)
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor")
@patch("mist.inference.inference_runners.get_strategy")
@patch("mist.inference.inference_runners.get_ensembler")
@patch("mist.inference.inference_runners.get_inferer")
@patch("mist.inference.inference_runners.model_loader.load_model_from_config")
@patch("mist.inference.inference_runners.dali_loader.get_test_dataset")
@patch(
    "mist.inference.inference_runners.ants.image_read",
    return_value=_DummyANTsImage()
)
@patch("mist.inference.inference_runners.utils.read_json_file")
def test_test_on_fold_success_no_crop_tta_enabled(
    mock_read_json,
    _mock_ants_read,
    mock_get_test_dataset,
    mock_load_model,
    mock_get_inferer,
    mock_get_ensembler,
    mock_get_strategy,
    mock_Predictor,
    mock_predict_single,
    mock_image_write,
    _mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """Test with one case, no cropping, TTA enabled, and writes output."""
    # Filesystem layout.
    results_dir, numpy_dir = _prep_dirs(tmp_path)
    fold = 0
    _make_train_df(fold).to_csv(results_dir / "train_paths.csv", index=False)
    _make_bbox_df().to_csv(results_dir / "fg_bboxes.csv", index=False)
    (results_dir / "models" / f"fold_{fold}.pt").write_bytes(b"\x00")

    # Config (use fixture directly).
    cfg = copy.deepcopy(mock_mist_config)
    cfg["preprocessing"]["crop_to_foreground"] = False
    cfg["inference"]["tta"]["enabled"] = True
    (results_dir / "config.json").write_text("{}", encoding="utf-8")
    mock_read_json.return_value = cfg

    # CLI-like args namespace.
    mist_args = SimpleNamespace(results=str(results_dir), numpy=str(numpy_dir))

    # DALI loader yields one item.
    mock_get_test_dataset.return_value = _DummyLoader(n=1)

    # Model stub.
    model = _DummyModel()
    mock_load_model.return_value = model

    # Inferer/ensembler/tta factories.
    mock_get_inferer.return_value = lambda **_: SimpleNamespace(name="inferer")
    mock_get_ensembler.return_value = SimpleNamespace(name="ensembler")
    mock_get_strategy.return_value = lambda: SimpleNamespace(name="tta")

    # Predictor stub (we don't need it to do anything since predict_single is
    # patched).
    mock_Predictor.return_value = MagicMock()

    # Predict returns an ANTs-like image.
    mock_predict_single.return_value = _DummyANTsImage()

    # ic constants for filtering patient columns.
    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(PATIENT_DF_IGNORED_COLUMNS={"id", "fold"}),
        raising=False,
    )
    # Force CPU fallback when device=None.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    ir.test_on_fold(mist_args=mist_args, fold_number=fold, device=None) # type: ignore

    # Assertions: model & dataset loaded, prediction written, TTA branch taken.
    mock_get_test_dataset.assert_called_once()
    mock_load_model.assert_called_once()
    mock_predict_single.assert_called_once()
    mock_image_write.assert_called_once()
    out_path = mock_image_write.call_args.args[1]
    assert out_path.endswith(
        os.path.join("predictions", "train", "raw", "p1.nii.gz")
    )
    assert any(
        c.args == (cfg["inference"]["tta"]["strategy"],)
        for c in mock_get_strategy.mock_calls
    )


@patch(
    "mist.inference.inference_runners.utils.get_progress_bar",
    return_value=_PB()
)
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor")
@patch("mist.inference.inference_runners.get_strategy")
@patch("mist.inference.inference_runners.get_ensembler")
@patch("mist.inference.inference_runners.get_inferer")
@patch("mist.inference.inference_runners.model_loader.load_model_from_config")
@patch("mist.inference.inference_runners.dali_loader.get_test_dataset")
@patch(
    "mist.inference.inference_runners.ants.image_read",
    return_value=_DummyANTsImage()
)
@patch("mist.inference.inference_runners.utils.read_json_file")
def test_test_on_fold_crop_to_foreground_bbox_passed(
    mock_read_json,
    _mock_ants_read,
    mock_get_test_dataset,
    mock_load_model,
    mock_get_inferer,
    mock_get_ensembler,
    mock_get_strategy,
    mock_Predictor,
    mock_predict_single,
    _mock_image_write,
    _mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """Test with crop_to_foreground=True, bbox passed to predict_single_example.

    When crop_to_foreground=True and skip=False, bbox from CSV is forwarded to
    predict_single_example.
    """
    results_dir, numpy_dir = _prep_dirs(tmp_path)
    fold = 2
    _make_train_df(fold).to_csv(results_dir / "train_paths.csv", index=False)
    bbox_df = _make_bbox_df()
    bbox_df.to_csv(results_dir / "fg_bboxes.csv", index=False)
    (results_dir / "models" / f"fold_{fold}.pt").write_bytes(b"\x00")

    cfg = copy.deepcopy(mock_mist_config)
    cfg["preprocessing"]["crop_to_foreground"] = True
    cfg["preprocessing"]["skip"] = False
    cfg["inference"]["tta"]["enabled"] = False  # hit 'none' strategy path
    (results_dir / "config.json").write_text("{}", encoding="utf-8")
    mock_read_json.return_value = cfg

    mist_args = SimpleNamespace(results=str(results_dir), numpy=str(numpy_dir))
    mock_get_test_dataset.return_value = _DummyLoader(n=1)

    model = _DummyModel()
    mock_load_model.return_value = model

    mock_get_inferer.return_value = lambda **_: SimpleNamespace(name="inferer")
    mock_get_ensembler.return_value = SimpleNamespace(name="ensembler")

    def _strategy(name):
        return (
            (lambda: SimpleNamespace(name="none"))
            if name == "none" else (lambda: SimpleNamespace(name=name))
        )
    mock_get_strategy.side_effect = _strategy

    captured = {}
    def _capture_kwargs(**kwargs):
        captured.update(kwargs)
        return _DummyANTsImage()
    mock_predict_single.side_effect = _capture_kwargs

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(PATIENT_DF_IGNORED_COLUMNS={"id", "fold"}),
        raising=False,
    )

    ir.test_on_fold(mist_args=mist_args, fold_number=fold, device="cpu") # type: ignore

    # Foreground bbox forwarded
    assert "foreground_bounding_box" in captured
    assert captured["foreground_bounding_box"] == bbox_df.iloc[0].to_dict()
    # 'none' strategy invoked at least once
    assert any(
        (c.args and c.args[0] == "none") for c in mock_get_strategy.mock_calls
    )


@patch(
    "mist.inference.inference_runners.utils.get_progress_bar",
    return_value=_PB()
)
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor")
@patch(
    "mist.inference.inference_runners.get_strategy",
    return_value=lambda: SimpleNamespace(name="tta")
)
@patch(
    "mist.inference.inference_runners.get_ensembler",
    return_value=SimpleNamespace(name="ensembler")
)
@patch(
    "mist.inference.inference_runners.get_inferer",
    return_value=(lambda **_: SimpleNamespace(name="inferer"))
)
@patch("mist.inference.inference_runners.model_loader.load_model_from_config")
@patch("mist.inference.inference_runners.dali_loader.get_test_dataset")
@patch(
    "mist.inference.inference_runners.ants.image_read",
    return_value=_DummyANTsImage()
)
@patch("mist.inference.inference_runners.utils.read_json_file")
def test_test_on_fold_logs_errors_and_continues(
    mock_read_json,
    _mock_ants_read,
    mock_get_test_dataset,
    mock_load_model,
    _mock_get_inferer,
    _mock_get_ensembler,
    _mock_get_strategy,
    mock_Predictor,
    mock_predict_single,
    mock_image_write,
    mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """If a case fails, it is logged and the rest continue to write outputs."""
    results_dir, numpy_dir = _prep_dirs(tmp_path)
    fold = 0
    # Two cases, one fails.
    # Make paths dataframe.
    two_case_df = pd.DataFrame(
        [
            {"id": "pA", "fold": fold, "image": "/tmp/pA.nii.gz"},
            {"id": "pB", "fold": fold, "image": "/tmp/pB.nii.gz"},
        ]
    )
    two_case_df.to_csv(results_dir / "train_paths.csv", index=False)

    # Make foreground bounding boxes CSV.
    new_bbox = {
        "id": "pB", "x0": 0, "x1": 1, "y0": 0, "y1": 1, "z0": 0, "z1": 1
    }
    pd.concat(
        [_make_bbox_df(), pd.DataFrame([new_bbox])], ignore_index=True
    ).to_csv(results_dir / "fg_bboxes.csv", index=False)
    (results_dir / "models" / f"fold_{fold}.pt").write_bytes(b"\x00")

    cfg = copy.deepcopy(mock_mist_config)
    cfg["preprocessing"]["crop_to_foreground"] = False
    (results_dir / "config.json").write_text("{}", encoding="utf-8")
    mock_read_json.return_value = cfg

    mist_args = SimpleNamespace(results=str(results_dir), numpy=str(numpy_dir))

    mock_get_test_dataset.return_value = _DummyLoader(n=2)
    model = _DummyModel()
    mock_load_model.return_value = model

    # First prediction raises, second succeeds.
    mock_predict_single.side_effect = [RuntimeError("boom"), _DummyANTsImage()]

    # Capture console prints.
    printed = []
    console = MagicMock()
    console.print.side_effect = lambda msg: printed.append(str(msg))
    mock_console_cls.return_value = console

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(PATIENT_DF_IGNORED_COLUMNS={"id", "fold"}),
        raising=False,
    )

    ir.test_on_fold(mist_args=mist_args, fold_number=fold, device="cpu") # type: ignore

    # One error logged, one image written (for pB).
    assert any("Prediction failed for pA" in m for m in printed)
    assert mock_image_write.call_count == 1
    out_path = mock_image_write.call_args.args[1]
    assert out_path.endswith(
        os.path.join("predictions", "train", "raw", "pB.nii.gz")
    )


@pytest.mark.parametrize(
    "cuda_available, explicit_device, expected",
    [
        (False, None, "cpu"),
        (True, None, "cuda"),
        (True, "cpu", "cpu"),
        (False, "cuda", "cuda"),
    ],
)
@patch(
    "mist.inference.inference_runners.utils.get_progress_bar",
    return_value=_PB()
)
@patch("mist.inference.inference_runners.rich.console.Console")
@patch(
    "mist.inference.inference_runners.predict_single_example",
    return_value=_DummyANTsImage()
)
@patch("mist.inference.inference_runners.Predictor")
@patch(
    "mist.inference.inference_runners.get_strategy",
    return_value=lambda: SimpleNamespace(name="tta")
)
@patch(
    "mist.inference.inference_runners.get_ensembler",
    return_value=SimpleNamespace(name="ensembler")
)
@patch(
    "mist.inference.inference_runners.get_inferer",
    return_value=(lambda **_: SimpleNamespace(name="inferer"))
)
@patch("mist.inference.inference_runners.model_loader.load_model_from_config")
@patch("mist.inference.inference_runners.dali_loader.get_test_dataset")
@patch(
    "mist.inference.inference_runners.ants.image_read",
    return_value=_DummyANTsImage()
)
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.utils.read_json_file")
def test_test_on_fold_device_resolution(
    mock_read_json,
    _mock_image_write,
    _mock_ants_read,
    mock_get_test_dataset,
    mock_load_model,
    _mock_get_inferer,
    _mock_get_ensembler,
    _mock_get_strategy,
    mock_Predictor,
    _mock_predict_single,
    _mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
    cuda_available,
    explicit_device,
    expected,
    noop_cuda_tensor_to,
):
    """Test device selection behavior.

    More specifically, test explicit device overrides; otherwise cuda to cpu
    fallback.
    """
    results_dir, numpy_dir = _prep_dirs(tmp_path)
    fold = 1
    _make_train_df(fold).to_csv(results_dir / "train_paths.csv", index=False)
    _make_bbox_df().to_csv(results_dir / "fg_bboxes.csv", index=False)
    (results_dir / "models" / f"fold_{fold}.pt").write_bytes(b"\x00")

    cfg = copy.deepcopy(mock_mist_config)
    (results_dir / "config.json").write_text("{}", encoding="utf-8")
    mock_read_json.return_value = cfg

    mist_args = SimpleNamespace(results=str(results_dir), numpy=str(numpy_dir))
    mock_get_test_dataset.return_value = _DummyLoader(n=1)

    model = _DummyModel()
    mock_load_model.return_value = model

    # Observe device passed into Predictor.
    observed = {}

    class _Pred:
        def __init__(self, **kwargs):
            observed["device"] = kwargs["device"]

    mock_Predictor.side_effect = _Pred

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(PATIENT_DF_IGNORED_COLUMNS={"id", "fold"}),
        raising=False
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    ir.test_on_fold(
        mist_args=mist_args, fold_number=fold, device=explicit_device
    )

    assert observed["device"] == expected


# Tests for infer_from_dataframe.
def _df_single_case(tmp_path: Path):
    """Minimal dataframe with a single patient id + image path."""
    return pd.DataFrame(
        [{"id": "p1", "image": str((tmp_path / "images" / "p1.nii.gz"))}]
    )


def _ensure_dir(p: Path):
    """Ensure directory exists, create if not."""
    p.mkdir(parents=True, exist_ok=True)
    return p


@patch(
    "mist.inference.inference_runners.utils.get_progress_bar",
    return_value=_PB()
)
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor")
@patch("mist.inference.inference_runners.get_strategy")
@patch("mist.inference.inference_runners.get_ensembler")
@patch("mist.inference.inference_runners.get_inferer")
@patch(
    "mist.inference.inference_runners.inference_utils.load_test_time_models"
)
@patch(
    "mist.inference.inference_runners.inference_utils.validate_inference_images"
)
@patch("mist.inference.inference_runners.preprocess.preprocess_example")
def test_infer_from_dataframe_success_preprocess_path_tta_enabled(
    mock_preprocess_example,
    mock_validate_images,
    mock_load_models,
    mock_get_inferer,
    mock_get_ensembler,
    mock_get_strategy,
    mock_Predictor,
    mock_predict_single,
    mock_image_write,
    _mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """Test preprocessing branch (skip=False), TTA enabled, writes output."""
    # Input dataframe.
    df = _df_single_case(tmp_path)
    out_dir = str(_ensure_dir(tmp_path / "out"))
    models_dir = str(_ensure_dir(tmp_path / "models"))

    # Config copy.
    cfg = copy.deepcopy(mock_mist_config)

    # ic constants used in tensor conversion
    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(
            NUMPY_TO_TORCH_TRANSPOSE_AXES=(0, 1, 2),
            NUMPY_TO_TORCH_EXPAND_DIMS_AXES=0,
        ),
        raising=False,
    )

    # Validate_inference_images -> (anchor_image, image_paths).
    anchor = _DummyANTsImage()
    mock_validate_images.return_value = (anchor, [df.iloc[0]["image"]])

    # preprocess_example returns dict with numpy image + bbox.
    img_np = np.zeros((2, 2, 2), dtype=np.float32)
    bbox = {"x0": 0, "x1": 1, "y0": 0, "y1": 1, "z0": 0, "z1": 1}
    mock_preprocess_example.return_value = {"image": img_np, "fg_bbox": bbox}

    # Models + predictor stack.
    mock_load_models.return_value = [MagicMock(name="model0")]
    mock_get_inferer.return_value = lambda **_: SimpleNamespace(name="inferer")
    mock_get_ensembler.return_value = SimpleNamespace(name="ensembler")
    mock_get_strategy.return_value = lambda: SimpleNamespace(name="tta")
    mock_Predictor.return_value = MagicMock()
    mock_predict_single.return_value = _DummyANTsImage()

    # Device fallback.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    ir.infer_from_dataframe(
        paths_dataframe=df,
        output_directory=out_dir,
        mist_configuration=cfg,
        models_directory=models_dir,
        postprocessing_strategy_filepath=None,
        device=None,
    )

    # Assertions.
    mock_load_models.assert_called_once()
    mock_validate_images.assert_called_once()
    mock_preprocess_example.assert_called_once()
    mock_predict_single.assert_called_once()
    mock_image_write.assert_called_once()
    out_path = mock_image_write.call_args.args[1]
    assert out_path.endswith(os.path.join("out", "p1.nii.gz"))
    # TTA path: ensure strategy from config was used.
    assert any(
        c.args == (cfg["inference"]["tta"]["strategy"],)
        for c in mock_get_strategy.mock_calls
    )


@patch(
    "mist.inference.inference_runners.utils.get_progress_bar",
    return_value=_PB()
)
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor")
@patch("mist.inference.inference_runners.get_strategy")
@patch("mist.inference.inference_runners.get_ensembler")
@patch("mist.inference.inference_runners.get_inferer")
@patch("mist.inference.inference_runners.inference_utils.load_test_time_models")
@patch(
    "mist.inference.inference_runners.inference_utils.validate_inference_images"
)
@patch("mist.inference.inference_runners.preprocess.preprocess_example")
def test_infer_from_dataframe_success_tta_disabled(
    mock_preprocess_example,
    mock_validate_images,
    mock_load_models,
    mock_get_inferer,
    mock_get_ensembler,
    mock_get_strategy,
    mock_Predictor,
    mock_predict_single,
    mock_image_write,
    _mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """Test with no preprocessing, TTA disabled, and writes output."""
    df = _df_single_case(tmp_path)
    out_dir = str(_ensure_dir(tmp_path / "out"))
    models_dir = str(_ensure_dir(tmp_path / "models"))

    # Change config to skip preprocessing and disable TTA.
    cfg = copy.deepcopy(mock_mist_config)
    cfg["inference"]["tta"]["enabled"] = False

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(
            NUMPY_TO_TORCH_TRANSPOSE_AXES=(0, 1, 2),
            NUMPY_TO_TORCH_EXPAND_DIMS_AXES=0,
        ),
        raising=False,
    )

    anchor = _DummyANTsImage()
    mock_validate_images.return_value = (anchor, [df.iloc[0]["image"]])

    # preprocess_example returns dict with numpy image + bbox.
    img_np = np.zeros((2, 2, 2), dtype=np.float32)
    bbox = {"x0": 0, "x1": 1, "y0": 0, "y1": 1, "z0": 0, "z1": 1}
    mock_preprocess_example.return_value = {"image": img_np, "fg_bbox": bbox}

    mock_load_models.return_value = [MagicMock(name="model0")]
    mock_get_inferer.return_value = lambda **_: SimpleNamespace(name="inferer")
    mock_get_ensembler.return_value = SimpleNamespace(name="ensembler")

    # Ensure 'none' strategy when TTA disabled
    def _strategy(name):
        return (
            (lambda: SimpleNamespace(name="none")) if name == "none"
            else (lambda: SimpleNamespace(name=name))
        )
    mock_get_strategy.side_effect = _strategy

    mock_Predictor.return_value = MagicMock()
    mock_predict_single.return_value = _DummyANTsImage()

    ir.infer_from_dataframe(
        paths_dataframe=df,
        output_directory=out_dir,
        mist_configuration=cfg,
        models_directory=models_dir,
        postprocessing_strategy_filepath=None,
        device="cpu",
    )

    mock_predict_single.assert_called_once()
    mock_image_write.assert_called_once()
    # Verify 'none' invoked.
    assert any(
        (c.args and c.args[0] == "none") for c in mock_get_strategy.mock_calls
    )


@patch(
    "mist.inference.inference_runners.utils.get_progress_bar",
    return_value=_PB()
)
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor")
@patch(
    "mist.inference.inference_runners.get_strategy",
    return_value=lambda: SimpleNamespace(name="tta")
)
@patch(
    "mist.inference.inference_runners.get_ensembler",
    return_value=SimpleNamespace(name="ensembler")
)
@patch(
    "mist.inference.inference_runners.get_inferer",
    return_value=(lambda **_: SimpleNamespace(name="inferer"))
)
@patch(
    "mist.inference.inference_runners.inference_utils.load_test_time_models"
)
@patch(
    "mist.inference.inference_runners.inference_utils.validate_inference_images"
)
@patch("mist.inference.inference_runners.preprocess.preprocess_example")
def test_infer_from_dataframe_postprocess_applied_and_messages_printed(
    mock_preprocess_example,
    mock_validate_images,
    mock_load_models,
    _mock_get_inferer,
    _mock_get_ensembler,
    _mock_get_strategy,
    mock_Predictor,
    mock_predict_single,
    mock_image_write,
    mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """Test with postprocessing and check messages are printed after run."""
    df = _df_single_case(tmp_path)
    out_dir = str(_ensure_dir(tmp_path / "out"))
    models_dir = str(_ensure_dir(tmp_path / "models"))
    post_file = tmp_path / "post.json"
    post_file.write_text("{}", encoding="utf-8")

    # Copy configuration.
    cfg = copy.deepcopy(mock_mist_config)

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(
            NUMPY_TO_TORCH_TRANSPOSE_AXES=(0, 1, 2),
            NUMPY_TO_TORCH_EXPAND_DIMS_AXES=0,
        ),
        raising=False,
    )

    anchor = _DummyANTsImage()
    mock_validate_images.return_value = (anchor, [df.iloc[0]["image"]])
    mock_preprocess_example.return_value = {
        "image": np.zeros((2, 2, 2), dtype=np.float32), "fg_bbox": None
    }

    mock_load_models.return_value = [MagicMock(name="model0")]
    mock_Predictor.return_value = MagicMock()
    mock_predict_single.return_value = _DummyANTsImage()

    # Mock Postprocessor and capture call.
    with patch("mist.inference.inference_runners.Postprocessor") as MockPP:
        pp = MagicMock()
        pp.apply_strategy_to_single_example.return_value = (
            _DummyANTsImage(), ["warn: something"]
        )
        MockPP.return_value = pp

        printed = []
        console = MagicMock()
        console.print.side_effect = lambda msg: printed.append(str(msg))
        mock_console_cls.return_value = console

        ir.infer_from_dataframe(
            paths_dataframe=df,
            output_directory=out_dir,
            mist_configuration=cfg,
            models_directory=models_dir,
            postprocessing_strategy_filepath=str(post_file),
            device="cpu",
        )

        # Ensure postprocessor constructed & applied
        MockPP.assert_called_once()
        pp.apply_strategy_to_single_example.assert_called_once()
        # Since messages existed, a summary header + message printed
        assert any(
            "Inference completed with the following messages:" in m
            for m in printed
        )
        assert any("warn: something" in m for m in printed)
        mock_image_write.assert_called_once()


@patch("mist.inference.inference_runners.inference_utils.load_test_time_models")
@patch("mist.inference.inference_runners.get_inferer")
@patch("mist.inference.inference_runners.get_ensembler")
@patch("mist.inference.inference_runners.get_strategy")
@patch("mist.inference.inference_runners.Predictor")
@patch(
    "mist.inference.inference_runners.utils.get_progress_bar",
    return_value=_PB()
)
@patch("mist.inference.inference_runners.rich.console.Console")
def test_infer_from_dataframe_postprocess_file_missing_raises(
    mock_console,
    mock_get_progress_bar,
    mock_predictor,
    mock_get_strategy,
    mock_get_ensembler,
    mock_get_inferer,
    mock_load_models,
    tmp_path,
    mock_mist_config
):
    """If postprocess strategy file path is provided but missing."""
    df = _df_single_case(tmp_path)
    out_dir = str(_ensure_dir(tmp_path / "out"))
    models_dir = str(_ensure_dir(tmp_path / "models"))
    missing_path = tmp_path / "nope.json"

    # Copy configuration.
    cfg = copy.deepcopy(mock_mist_config)

    with pytest.raises(FileNotFoundError):
        ir.infer_from_dataframe(
            paths_dataframe=df,
            output_directory=out_dir,
            mist_configuration=cfg,
            models_directory=models_dir,
            postprocessing_strategy_filepath=str(missing_path),
            device="cpu",
        )


@patch(
    "mist.inference.inference_runners.utils.get_progress_bar",
    return_value=_PB()
)
@patch("mist.inference.inference_runners.rich.console.Console")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Predictor")
@patch(
    "mist.inference.inference_runners.get_strategy",
    return_value=lambda: SimpleNamespace(name="tta")
)
@patch(
    "mist.inference.inference_runners.get_ensembler",
    return_value=SimpleNamespace(name="ensembler")
)
@patch(
    "mist.inference.inference_runners.get_inferer",
    return_value=(lambda **_: SimpleNamespace(name="inferer"))
)
@patch("mist.inference.inference_runners.inference_utils.load_test_time_models")
@patch(
    "mist.inference.inference_runners.inference_utils.validate_inference_images"
)
@patch("mist.inference.inference_runners.preprocess.preprocess_example")
def test_infer_from_dataframe_logs_errors_and_continues_then_summarizes(
    mock_preprocess_example,
    mock_validate_images,
    mock_load_models,
    _mock_get_inferer,
    _mock_get_ensembler,
    _mock_get_strategy,
    mock_Predictor,
    mock_predict_single,
    mock_image_write,
    mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
):
    """First patient fails; second succeeds; prints error summary."""
    # Set up scenario with two patients, one fails.
    df = pd.DataFrame(
        [
            {"id": "pA", "image": str((tmp_path / "images" / "pA.nii.gz"))},
            {"id": "pB", "image": str((tmp_path / "images" / "pB.nii.gz"))},
        ]
    )
    out_dir = str(_ensure_dir(tmp_path / "out"))
    models_dir = str(_ensure_dir(tmp_path / "models"))
    cfg = copy.deepcopy(mock_mist_config)

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(
            NUMPY_TO_TORCH_TRANSPOSE_AXES=(0, 1, 2),
            NUMPY_TO_TORCH_EXPAND_DIMS_AXES=0,
        ),
        raising=False,
    )

    anchor = _DummyANTsImage()
    # Validate returns per-patient; we can just have it return same anchor;
    # function pulls by row.
    mock_validate_images.return_value = (anchor, [df.iloc[0]["image"]])

    mock_preprocess_example.return_value = {
        "image": np.zeros((2, 2, 2), dtype=np.float32), "fg_bbox": None
    }
    mock_load_models.return_value = [MagicMock(name="model0")]

    # First call raises, second succeeds.
    mock_predict_single.side_effect = [RuntimeError("boom"), _DummyANTsImage()]

    printed = []
    console = MagicMock()
    console.print.side_effect = lambda msg: printed.append(str(msg))
    mock_console_cls.return_value = console

    ir.infer_from_dataframe(
        paths_dataframe=df,
        output_directory=out_dir,
        mist_configuration=cfg,
        models_directory=models_dir,
        postprocessing_strategy_filepath=None,
        device="cpu",
    )

    # One error reported in summary, one image written.
    assert any("Prediction failed for pA" in m for m in printed)
    assert any(
        "Inference completed with the following messages:" in m
        for m in printed
    )
    assert mock_image_write.call_count == 1
    out_path = mock_image_write.call_args.args[1]
    assert out_path.endswith(os.path.join("out", "pB.nii.gz"))


@pytest.mark.parametrize(
    "cuda_available, explicit_device, expected",
    [
        (False, None, "cpu"),
        (True, None, "cuda"),
        (True, "cpu", "cpu"),
        (False, "cuda", "cuda"),
    ],
)
@patch(
    "mist.inference.inference_runners.utils.get_progress_bar",
    return_value=_PB()
)
@patch("mist.inference.inference_runners.rich.console.Console")
@patch(
    "mist.inference.inference_runners.predict_single_example",
    return_value=_DummyANTsImage()
)
@patch("mist.inference.inference_runners.Predictor")
@patch(
    "mist.inference.inference_runners.get_strategy",
    return_value=lambda: SimpleNamespace(name="tta")
)
@patch(
    "mist.inference.inference_runners.get_ensembler",
    return_value=SimpleNamespace(name="ensembler")
)
@patch(
    "mist.inference.inference_runners.get_inferer",
    return_value=lambda **_: SimpleNamespace(name="inferer")
)
@patch(
    "mist.inference.inference_runners.inference_utils.load_test_time_models",
    return_value=[MagicMock(name="model0")]
)
@patch(
    "mist.inference.inference_runners.inference_utils.validate_inference_images",
    return_value=(_DummyANTsImage(), ["X"])
)
@patch(
    "mist.inference.inference_runners.preprocess.preprocess_example",
    return_value={
        "image": np.zeros((2, 2, 2), dtype=np.float32), "fg_bbox": None
    }
)
@patch("mist.inference.inference_runners.ants.image_write")
def test_infer_from_dataframe_device_resolution(
    _mock_image_write,
    _mock_preprocess_example,
    _mock_validate_images,
    _mock_load_models,
    _mock_get_inferer,
    _mock_get_ensembler,
    _mock_get_strategy,
    mock_Predictor,
    _mock_predict_single,
    _mock_console_cls,
    _mock_pb,
    tmp_path,
    mock_mist_config,
    monkeypatch,
    cuda_available,
    explicit_device,
    expected,
    noop_cuda_tensor_to,
):
    """Test device selection behavior in infer_from_dataframe."""
    df = _df_single_case(tmp_path)
    out_dir = str(_ensure_dir(tmp_path / "out"))
    models_dir = str(_ensure_dir(tmp_path / "models"))

    # Copy configuration.
    cfg = copy.deepcopy(mock_mist_config)

    monkeypatch.setattr(
        ir,
        "ic",
        SimpleNamespace(
            NUMPY_TO_TORCH_TRANSPOSE_AXES=(0, 1, 2),
            NUMPY_TO_TORCH_EXPAND_DIMS_AXES=0,
        ),
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    observed = {}

    class _Pred:
        def __init__(self, **kwargs):
            observed["device"] = kwargs["device"]

    mock_Predictor.side_effect = _Pred

    ir.infer_from_dataframe(
        paths_dataframe=df,
        output_directory=out_dir,
        mist_configuration=cfg,
        models_directory=models_dir,
        postprocessing_strategy_filepath=None,
        device=explicit_device,
    )

    assert observed["device"] == expected
