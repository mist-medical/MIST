"""Unit tests for the BaseTrainer implementation."""
import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
import pandas as pd
import pytest
import torch
from torch import nn

# MIST imports.
from mist.training.trainers import base_trainer as bt

# Setup base trainer for tests.
BaseTrainer = bt.BaseTrainer


class DummyIter:
    """DALI-style loader with .next()[0] and .reset()."""

    def __init__(self, batch, length: int):
        """Initialize the iterator.

        Args:
            batch: The fixed batch to emit.
            length: Number of steps before wrapping.
        """
        self._batch = batch
        self._length = max(0, length)
        self._i = 0

    def next(self):
        """Return a tuple(batch,) each call, wrapping after length steps."""
        if self._i >= self._length:
            # Emulate stop by wrapping around (trainer controls step count).
            self._i = 0
        self._i += 1
        return (self._batch,)

    def reset(self):
        """Reset the internal step counter."""
        self._i = 0


class DummyModel(nn.Module):
    """Minimal neural network for testing."""

    def __init__(self):
        """Initialize the dummy model."""
        super().__init__()
        self.l = nn.Linear(2, 2)

    def forward(self, x):
        """Forward pass (unused in tests, defined for completeness)."""
        return self.l(x)


class DummyDDP(nn.Module):
    """Minimal DDP stand-in with .module for checkpoint friendliness."""

    def __init__(self, module, device_ids=None, find_unused_parameters=False):
        """Initialize the dummy DDP wrapper."""
        super().__init__()
        self.module = module

    def forward(self, *a, **k):
        """Forward pass that delegates to the wrapped module."""
        return self.module(*a, **k)


class DummySummaryWriter:
    """In-memory SummaryWriter stand-in that records scalars."""

    def __init__(self, log_dir):
        """Initialize the writer with a given log directory."""
        self.log_dir = log_dir
        self.scalars = []
        self.closed = False

    def add_scalars(self, tag, scalars, step):
        """Record a scalar dictionary at a given step."""
        self.scalars.append((tag, dict(scalars), int(step)))

    def flush(self):
        """No-op flush."""
        pass

    def close(self):
        """Mark the writer as closed."""
        self.closed = True


class DummyProgressCtx:
    """Context manager used by train/val progress bars with .update()."""

    def __init__(self, *a, **k):
        """Initialize the dummy context manager."""
        pass

    def __enter__(self):
        """Enter the context."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit the context."""
        return False

    def update(self, **kwargs):
        """Accept updates for loss/lr without side effects."""
        # Accept loss/lr updates; no-op.
        pass


class DummyTrainer(BaseTrainer):
    """Concrete subclass providing minimal train/val steps and loaders."""

    def __init__(self, mist_args, train_loss_value=1.0, val_loss_value=2.0):
        """Initialize the dummy trainer with fixed loss values."""
        self._train_loss_value = train_loss_value
        self._val_loss_value = val_loss_value
        super().__init__(mist_args)

    def build_dataloaders(self, fold_data, rank, world_size):
        """Build deterministic DALI-style loaders with a fixed batch tensor."""
        # Build deterministic DALI-style loaders with a fixed batch tensor.
        train_len = int(fold_data["steps_per_epoch"])
        val_len = max(1, len(fold_data["val_images"]) // max(1, world_size))
        batch = torch.zeros(1, 2)
        return DummyIter(batch, train_len), DummyIter(batch, val_len)

    def training_step(self, **kwargs):
        """Return a scalar tensor for the training loss."""
        state = kwargs["state"]
        dev = next(state["model"].parameters()).device
        return torch.tensor(self._train_loss_value, device=dev)

    def validation_step(self, **kwargs):
        """Return a scalar tensor for the validation loss."""
        state = kwargs["state"]
        dev = next(state["model"].parameters()).device
        return torch.tensor(self._val_loss_value, device=dev)


@pytest.fixture(autouse=True)
def patch_cuda_and_moves(monkeypatch):
    """Patch CUDA availability and device moves so tests run CPU-only."""
    # Pretend CUDA exists so the trainer doesn't raise in
    # _update_num_gpus_in_config.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda idx: None, raising=False
    )

    # Make .to(device) a no-op so models stay on CPU.
    monkeypatch.setattr(
        nn.Module, "to", lambda self, *a, **k: self, raising=False
    )

    # Inside the module under test, make torch.tensor(...) ignore device kwarg.
    _orig_tensor = torch.tensor

    def _cpu_tensor(*args, **kwargs):
        """Create a CPU tensor regardless of any device kwarg."""
        kwargs.pop("device", None)  # Strip any device='cuda:0' etc.
        return _orig_tensor(*args, **kwargs)

    monkeypatch.setattr(bt.torch, "tensor", _cpu_tensor, raising=False)


@pytest.fixture
def tmp_pipeline(tmp_path: Path) -> Tuple[Path, Path]:
    """Create a temporary results and numpy directory with required files."""
    results = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    (results / "models").mkdir(parents=True)
    (results / "logs").mkdir(parents=True)
    numpy_dir.mkdir(parents=True)

    # Minimal train_paths.csv with fold assignments.
    df = pd.DataFrame({"id": ["p0", "p1", "p2", "p3"], "fold": [0, 1, 0, 1]})
    df.to_csv(results / "train_paths.csv", index=False)

    # Minimal config.json matching the production structure.
    config = {
        "model": {"architecture": "dummy", "params": {}},
        "training": {
            "nfolds": 2,
            "folds": [0],
            "epochs": 1,
            "batch_size_per_gpu": 1,
            "hardware": {
                "master_addr": "127.0.0.1",
                "master_port": 12355,
                "communication_backend": "nccl",
                "num_gpus": 1,  # Will be overwritten.
            },
            "min_steps_per_epoch": 2,
            "val_percent": 0.0,
            "seed": 0,
            "loss": {
                "name": "dummy_loss",
                "params": {
                    "use_dtms": False,
                    "composite_loss_weighting": None,
                },
            },
            "optimizer": "sgd",
            "l2_penalty": 0.0,
            "lr_scheduler": "constant",
            "amp": False,
        },
    }
    (results / "config.json").write_text(json.dumps(config))

    # Numpy layout is mocked by tests via utils.get_numpy_file_paths_list.
    return results, numpy_dir


@pytest.fixture
def mist_args(tmp_pipeline):
    """Build a CLI-like args namespace used by the trainer."""
    results, numpy_dir = tmp_pipeline
    return SimpleNamespace(
        results=str(results),
        numpy=str(numpy_dir),
        model=None,
        patch_size=None,
        pocket=False,
        folds=None,
        epochs=None,
        batch_size_per_gpu=None,
        loss=None,
        use_dtms=False,
        composite_loss_weighting=None,
        optimizer=None,
        l2_penalty=None,
        learning_rate=None,
        lr_scheduler=None,
    )


@pytest.fixture(autouse=True)
def patch_utils(monkeypatch):
    """Patch utils helpers to avoid touching the filesystem."""

    def fake_get_paths(base_dir, folder, patient_ids):
        """Return a list of pseudo-paths for trainer logic."""
        return [f"{folder}/{pid}.npy" for pid in patient_ids]

    monkeypatch.setattr(bt.utils, "get_numpy_file_paths_list", fake_get_paths)
    # JSON read/write pass-through to real file IO is fine here.


@pytest.fixture(autouse=True)
def patch_registries(monkeypatch):
    """Patch registries: model, loss, optimizer, and LR scheduler."""
    # get_model -> DummyModel.
    monkeypatch.setattr(bt, "get_model", lambda arch, **p: DummyModel())

    # get_loss -> simple placeholder (unused by training_step here).
    monkeypatch.setattr(bt, "get_loss", lambda name: object())

    # alpha scheduler registry -> dummy object.
    monkeypatch.setattr(bt, "get_alpha_scheduler", lambda cfg: object())

    # optimizer -> SGD with optional lr and weight_decay.
    def fake_get_optimizer(
        name, params, weight_decay, eps, lr=None, l2_penalty=None
    ):
        """Build a simple SGD optimizer for the provided parameters."""
        lr = 0.1 if lr is None else float(lr)
        wd = float(weight_decay if l2_penalty is None else l2_penalty)
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)

    monkeypatch.setattr(bt, "get_optimizer", fake_get_optimizer)

    # lr scheduler -> a scheduler with .step().
    class DummyScheduler:
        """No-op scheduler with a step method."""

        def step(self):
            """No-op scheduler step."""
            pass

    monkeypatch.setattr(
        bt,
        "get_lr_scheduler",
        lambda name, optimizer, epochs: DummyScheduler(),
    )


@pytest.fixture(autouse=True)
def patch_ddp_and_tb_and_save(monkeypatch):
    """Patch DDP, TensorBoard SummaryWriter, progress bars, and torch.save."""
    # Patch the DDP symbol used inside the module.
    monkeypatch.setattr(bt, "DDP", DummyDDP)
    # SummaryWriter -> dummy collector.
    monkeypatch.setattr(bt, "SummaryWriter", DummySummaryWriter)
    # Progress bars -> contexts with update().
    monkeypatch.setattr(bt.progress_bar, "TrainProgressBar", DummyProgressCtx)
    monkeypatch.setattr(
        bt.progress_bar, "ValidationProgressBar", DummyProgressCtx
    )
    # torch.save -> no-op.
    monkeypatch.setattr(torch, "save", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def patch_dist(monkeypatch):
    """Patch torch.distributed to be inert but count calls."""
    calls = {
        "init": 0,
        "destroy": 0,
        "all_reduce": 0,
        "broadcast": 0,
        "barrier": 0,
    }

    class FakeDist:
        """Inert torch.distributed replacement for testing."""

        _initialized = False

        @staticmethod
        def is_initialized():
            """Return whether the fake process group is initialized."""
            return FakeDist._initialized

        @staticmethod
        def init_process_group(backend, rank, world_size):
            """Initialize the fake process group."""
            calls["init"] += 1
            FakeDist._initialized = True

        @staticmethod
        def destroy_process_group():
            """Destroy the fake process group."""
            calls["destroy"] += 1
            FakeDist._initialized = False

        @staticmethod
        def all_reduce(t, op=None):
            """No-op all_reduce that only counts calls."""
            calls["all_reduce"] += 1

        @staticmethod
        def broadcast(t, src):
            """No-op broadcast that only counts calls."""
            calls["broadcast"] += 1

        @staticmethod
        def barrier():
            """No-op barrier that only counts calls."""
            calls["barrier"] += 1

        ReduceOp = SimpleNamespace(SUM=0)

    # Patch the torch.distributed used inside the module.
    monkeypatch.setattr(bt, "dist", FakeDist)
    return calls


def test_update_num_gpus_and_batchsize(tmp_pipeline, mist_args, monkeypatch):
    """Ensure GPU count is written to config and batch size is computed."""
    # Simulate single GPU.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    trainer = DummyTrainer(mist_args)
    assert trainer.config["training"]["hardware"]["num_gpus"] == 1
    assert (
        trainer.batch_size
        == trainer.config["training"]["batch_size_per_gpu"] * 1
    )


def test_setup_folds_no_valsplit(tmp_pipeline, mist_args, monkeypatch):
    """Verify fold setup without validation split and DTMs."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    trainer = DummyTrainer(mist_args)
    fold0 = trainer.folds[0]
    # steps_per_epoch = min(min_steps, len(train_images) // batch).
    assert fold0["steps_per_epoch"] == min(
        trainer.config["training"]["min_steps_per_epoch"],
        len(fold0["train_images"]) // trainer.batch_size,
    )
    # DTMs disabled by default.
    assert fold0["train_dtms"] is None


def test_setup_folds_with_dtms_and_valsplit(
    tmp_pipeline, mist_args, monkeypatch
):
    """Verify fold setup when DTMs and validation split are enabled."""
    # Enable DTMs and validation split in config.json.
    results, _ = tmp_pipeline
    cfg = json.loads((Path(results) / "config.json").read_text())
    cfg["training"]["loss"]["params"]["use_dtms"] = True
    cfg["training"]["val_percent"] = 0.5
    (Path(results) / "config.json").write_text(json.dumps(cfg))

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    trainer = DummyTrainer(mist_args)
    fold0 = trainer.folds[0]
    assert isinstance(fold0["train_dtms"], list)
    # With 50% split, we should have moved some items from train to val.
    assert len(fold0["val_images"]) > 0


def test_build_components_single_gpu(tmp_pipeline, mist_args, monkeypatch):
    """Ensure build_components returns a non-DDP model on single GPU."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=1)
    # Not wrapped in DDP if world_size == 1 (patched DDP, but code guards it).
    assert not isinstance(state["model"], DummyDDP)
    # Scaler respects AMP flag.
    assert state["scaler"].is_enabled() is False


def test_build_components_multi_gpu_wraps_with_ddp(
    tmp_pipeline, mist_args, monkeypatch
):
    """Ensure build_components wraps the model with DDP on multi-GPU."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    # Flip AMP on.
    results, _ = tmp_pipeline
    cfg = json.loads((Path(results) / "config.json").read_text())
    cfg["training"]["amp"] = True
    (Path(results) / "config.json").write_text(json.dumps(cfg))

    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=2)
    assert isinstance(state["model"], DummyDDP)
    assert state["scaler"].is_enabled() is True


def test_setup_initializes_process_group_once(
    tmp_pipeline, mist_args, monkeypatch, patch_dist
):
    """Ensure setup initializes the process group only once."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    trainer = DummyTrainer(mist_args)
    trainer.setup(rank=0, world_size=2)
    trainer.setup(rank=0, world_size=2)  # Second call should be no-op.
    assert patch_dist["init"] == 1  # Only initialized once.


def test_train_fold_runs_full_epoch(
    tmp_pipeline, mist_args, monkeypatch, patch_dist
):
    """Run a full epoch on rank 0 and verify collectives are called."""
    # world_size=1 simplifies collectives.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)

    # Ensure enough val samples so val_steps > 0 (true with 2 samples/fold).
    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=2.0)
    trainer.train_fold(fold=0, rank=0, world_size=1)

    # Collectives were called and should not error.
    assert patch_dist["broadcast"] >= 1
    assert patch_dist["barrier"] >= 2  # End of train/val loop barriers.


def test_train_fold_early_stop_on_nan(
    tmp_pipeline, mist_args, monkeypatch, patch_dist
):
    """NaN training loss should trigger early stop and cleanup."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    trainer = DummyTrainer(
        mist_args, train_loss_value=float("nan"), val_loss_value=2.0
    )
    trainer.train_fold(fold=0, rank=0, world_size=1)
    # We should have called destroy_process_group through cleanup().
    assert patch_dist["destroy"] >= 1


def test_overwrite_config_from_args(tmp_pipeline, mist_args, monkeypatch):
    """Verify all CLI overrides are reflected in the trainer's config."""
    # Provide some CLI overrides.
    mist_args.model = "myarch"
    mist_args.patch_size = [32, 32, 32]
    mist_args.folds = [0]
    mist_args.epochs = 3
    mist_args.batch_size_per_gpu = 2
    mist_args.loss = "my_loss"
    mist_args.use_dtms = True
    mist_args.composite_loss_weighting = "linear"
    mist_args.optimizer = "adamw"
    mist_args.l2_penalty = 0.01
    mist_args.learning_rate = 0.005
    mist_args.lr_scheduler = "cosine"
    mist_args.pocket = True

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    trainer = DummyTrainer(mist_args)

    cfg = trainer.config
    assert cfg["model"]["architecture"] == "myarch"
    assert cfg["model"]["params"]["patch_size"] == [32, 32, 32]
    assert cfg["model"]["params"]["use_pocket_model"] is True
    assert cfg["training"]["epochs"] == 3
    assert cfg["training"]["batch_size_per_gpu"] == 2
    assert cfg["training"]["loss"]["name"] == "my_loss"
    assert cfg["training"]["loss"]["params"]["use_dtms"] is True
    assert cfg["training"]["loss"]["composite_loss_weighting"] == "linear"
    assert cfg["training"]["optimizer"] == "adamw"
    assert cfg["training"]["l2_penalty"] == pytest.approx(0.01)
    assert cfg["training"]["learning_rate"] == pytest.approx(0.005)
    assert cfg["training"]["lr_scheduler"] == "cosine"


def test_fit_single_gpu_calls_run_directly(
    tmp_pipeline, mist_args, monkeypatch
):
    """fit() should call run_cross_validation directly on single GPU."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    called = {"run": 0}

    def spy_run(rank, world_size):
        """Spy that records run_cross_validation invocations."""
        assert rank == 0 and world_size == 1
        called["run"] += 1

    trainer = DummyTrainer(mist_args)
    trainer.run_cross_validation = spy_run  # type: ignore
    trainer.fit()
    assert called["run"] == 1


def test_fit_multi_gpu_uses_spawn(tmp_pipeline, mist_args, monkeypatch):
    """fit() should use mp.spawn when multiple GPUs are available."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    spawned = {"count": 0}

    def fake_spawn(fn, args, nprocs, join):
        """Fake mp.spawn that validates arguments and counts invocations."""
        assert nprocs == 2
        assert isinstance(args, tuple) and args[0] == 2
        spawned["count"] += 1

    monkeypatch.setattr(bt.mp, "spawn", fake_spawn)
    trainer = DummyTrainer(mist_args)
    trainer.fit()
    assert spawned["count"] == 1


def test_invalid_folds_subset_raises(tmp_pipeline, mist_args):
    """Invalid folds in CLI args should raise during trainer construction.

    If mist_args.folds includes an index >= nfolds, _overwrite_config_from_args
    should raise ValueError during BaseTrainer.__init__.
    """
    results, _ = tmp_pipeline

    # Make valid folds be {0, 1}; any '2' should be invalid.
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["nfolds"] = 2
    cfg_path.write_text(json.dumps(cfg))

    # Ask for an out-of-range fold.
    mist_args.folds = [0, 2]

    # Constructing DummyTrainer calls __init__ -> _overwrite_config_from_args,
    # which should raise on the invalid folds subset.
    with pytest.raises(ValueError) as excinfo:
        DummyTrainer(mist_args)

    msg = str(excinfo.value)
    assert "subset of [0, 1, ..., nfolds-1]" in msg
    assert "Found folds: [0, 2]" in msg


def test_update_num_gpus_raises_when_cuda_unavailable(
    tmp_pipeline, mist_args, monkeypatch
):
    """_update_num_gpus_in_config should fail fast when CUDA is unavailable."""
    # Force CUDA unavailable.
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: False, raising=False
    )

    with pytest.raises(ValueError) as excinfo:
        DummyTrainer(mist_args)  # __init__ calls _update_num_gpus_in_config.

    msg = str(excinfo.value)
    assert "CUDA is not available" in msg


def test_update_num_gpus_raises_when_zero_devices(
    tmp_pipeline, mist_args, monkeypatch
):
    """If CUDA is available but device_count() == 0, raise a clear error."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0, raising=False)

    with pytest.raises(ValueError) as excinfo:
        DummyTrainer(mist_args)  # __init__ calls _update_num_gpus_in_config.

    msg = str(excinfo.value)
    assert "device_count() == 0" in msg
    assert "CUDA_VISIBLE_DEVICES" in msg  # Sanity check message contents.


def test_update_num_gpus_sets_config_and_persists(
    tmp_pipeline, mist_args, monkeypatch
):
    """When CUDA is available, write the detected GPU count to config."""
    # Pretend we have 2 GPUs.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)

    # Construct trainer (triggers _update_num_gpus_in_config in __init__).
    results, _ = tmp_pipeline
    trainer = DummyTrainer(mist_args)

    # In-memory config updated.
    assert trainer.config["training"]["hardware"]["num_gpus"] == 2

    # And persisted to config.json.
    cfg_path = Path(results) / "config.json"
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["training"]["hardware"]["num_gpus"] == 2


def test_train_fold_raises_when_val_images_less_than_world_size(
    tmp_pipeline, mist_args, monkeypatch
):
    """Raise when validation set size is less than the number of GPUs."""
    results, _ = tmp_pipeline
    trainer = DummyTrainer(mist_args)

    # Patch setup to skip DDP init.
    monkeypatch.setattr(trainer, "setup", lambda *a, **k: None)
    # Patch build_components to skip model creation.
    monkeypatch.setattr(trainer, "build_components", lambda *a, **k: None)

    # Minimal fold data: only 1 validation image, but 2 GPUs.
    trainer.folds = [
        {
            "val_images": ["val_img_0"],  # Length 1.
            "train_images": ["train_img_0", "train_img_1"],
            "steps_per_epoch": 1,
        }
    ]

    with pytest.raises(ValueError) as excinfo:
        trainer.train_fold(fold=0, rank=0, world_size=2)

    msg = str(excinfo.value)
    assert "Not enough validation data" in msg
    assert "reduce the number of GPUs" in msg


def test_train_loop_else_branch_rank_nonzero(
    tmp_pipeline, mist_args, monkeypatch, patch_dist
):
    """Cover the non-rank-0 training loop (else branch) and collectives."""
    # Pretend we have 2 GPUs but keep everything CPU-only in tests.
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(
        torch.cuda, "device_count", lambda: 2, raising=False
    )
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda idx: None, raising=False
    )

    # Make Module.to(...) a no-op so the model stays on CPU.
    monkeypatch.setattr(
        nn.Module, "to", lambda self, *a, **k: self, raising=False
    )

    # Build the trainer.
    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=2.0)

    # Ensure fold 0 has >= 1 val step when world_size=2 and a few train steps.
    # steps_per_epoch = 2 -> two training iterations on rank != 0.
    # len(val_images) = 2 -> val_steps = 2 // 2 = 1.
    trainer.folds[0]["steps_per_epoch"] = 2
    trainer.folds[0]["val_images"] = ["val0", "val1"]
    # Keep some train images (not actually consumed by DummyIter logic).
    trainer.folds[0]["train_images"] = ["tr0", "tr1", "tr2"]

    # Avoid initializing a real process group (patched dist is inert anyway).
    monkeypatch.setattr(trainer, "setup", lambda *a, **k: None)

    # Run one full epoch on rank 1 (non-zero), world_size=2 to hit the else
    # branch.
    trainer.train_fold(fold=0, rank=1, world_size=2)

    # In the else branch, we all_reduce once per train step and once per val
    # step. With steps_per_epoch=2 and val_steps=1, expect at least 3 calls.
    assert patch_dist["all_reduce"] >= 3

    # Sanity: broadcast/barrier should be called too (no strict counts).
    assert patch_dist["broadcast"] >= 1
    assert patch_dist["barrier"] >= 1


def test_validation_else_branch_rank_nonzero(
    tmp_pipeline, mist_args, monkeypatch, patch_dist
):
    """Cover the non-rank-0 validation loop (else branch)."""
    # Pretend we have 2 GPUs, but keep all ops on CPU for tests.
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: True, raising=False
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda idx: None, raising=False
    )
    monkeypatch.setattr(
        nn.Module, "to", lambda self, *a, **k: self, raising=False
    )

    trainer = DummyTrainer(mist_args, train_loss_value=1.0, val_loss_value=2.0)

    # Ensure no training iterations and exactly one validation step:
    # val_steps = len(val_images) // world_size = 2 // 2 = 1.
    trainer.folds[0]["steps_per_epoch"] = 0
    trainer.folds[0]["val_images"] = ["val0", "val1"]  # Length 2.
    trainer.folds[0]["train_images"] = ["tr0", "tr1"]  # Not used.

    # Avoid real process-group init; our patched dist is inert.
    monkeypatch.setattr(trainer, "setup", lambda *a, **k: None)

    # Track all_reduce calls before/after to prove the else-branch ran.
    before = patch_dist["all_reduce"]

    # Run with rank=1 (non-zero) to take the validation else-branch.
    trainer.train_fold(fold=0, rank=1, world_size=2)

    after = patch_dist["all_reduce"]
    # Expect at least one all_reduce from the non-zero rank validation loop.
    assert (after - before) >= 1

    # Sanity: we should have broadcast/barrier during the epoch lifecycle.
    assert patch_dist["broadcast"] >= 1
    assert patch_dist["barrier"] >= 1


def test_validation_no_improvement_message(
    tmp_pipeline, mist_args, monkeypatch
):
    """Print a message when validation loss does not improve on rank 0."""
    # Keep CPU-only in tests.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda idx: None, raising=False
    )
    monkeypatch.setattr(
        nn.Module, "to", lambda self, *a, **k: self, raising=False
    )

    # Set val_loss_value high enough to avoid improvement.
    trainer = DummyTrainer(
        mist_args, train_loss_value=1.0, val_loss_value=float("inf")
    )

    # Force one train step and at least one val step.
    trainer.folds[0]["steps_per_epoch"] = 1
    # world_size=1 -> val_steps=2.
    trainer.folds[0]["val_images"] = ["val0", "val1"]

    # Patch setup to skip DDP.
    monkeypatch.setattr(trainer, "setup", lambda *a, **k: None)

    # Capture console output.
    buf = StringIO()
    monkeypatch.setattr(
        trainer.console, "print", lambda msg: buf.write(str(msg))
    )

    # Run training on rank=0 to hit rank-0 validation branch.
    trainer.train_fold(fold=0, rank=0, world_size=1)

    # Check that the message was printed.
    output = buf.getvalue()
    assert "Validation loss did not improve" in output


def test_run_cross_validation_rank0_prints_and_calls_all_folds(
    tmp_pipeline, mist_args, monkeypatch
):
    """Rank 0 should print start message and call train_fold for each fold."""
    results, _ = tmp_pipeline

    # Configure multiple folds to ensure iteration.
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["folds"] = [0, 1]
    cfg_path.write_text(json.dumps(cfg))

    trainer = DummyTrainer(mist_args)

    # Capture console output.
    out = []
    monkeypatch.setattr(
        trainer.console, "print", lambda msg: out.append(str(msg))
    )

    # Spy on train_fold.
    calls = []

    def spy_train_fold(fold, rank, world_size):
        """Record invocations of train_fold."""
        calls.append((fold, rank, world_size))

    monkeypatch.setattr(trainer, "train_fold", spy_train_fold)

    # Run as rank 0.
    trainer.run_cross_validation(rank=0, world_size=2)

    # Message printed once.
    assert any("Starting training" in s for s in out)
    # train_fold called for each fold with correct args.
    assert calls == [(0, 0, 2), (1, 0, 2)]


def test_run_cross_validation_nonzero_rank_no_print_but_calls_folds(
    tmp_pipeline, mist_args, monkeypatch
):
    """Non-zero rank should not print, but still call train_fold."""
    results, _ = tmp_pipeline

    # Ensure multiple folds again.
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["folds"] = [0, 1]
    cfg_path.write_text(json.dumps(cfg))

    trainer = DummyTrainer(mist_args)

    # Capture console output (should remain empty).
    out = []
    monkeypatch.setattr(
        trainer.console, "print", lambda msg: out.append(str(msg))
    )

    # Spy on train_fold.
    calls = []

    def spy_train_fold(fold, rank, world_size):
        """Record invocations of train_fold."""
        calls.append((fold, rank, world_size))

    monkeypatch.setattr(trainer, "train_fold", spy_train_fold)

    # Run as a non-zero rank.
    trainer.run_cross_validation(rank=1, world_size=2)

    # No "Starting training" print on non-zero ranks.
    assert not any("Starting training" in s for s in out)
    # But train_fold still called for each fold with the non-zero rank.
    assert calls == [(0, 1, 2), (1, 1, 2)]


@pytest.mark.parametrize("schedule_cfg", [None, "linear"])
def test_build_components_composite_loss_scheduler(
    tmp_pipeline, mist_args, monkeypatch, schedule_cfg
):
    """Verify composite-loss alpha scheduler is created only when configured.

    When `training["loss"]["params"]["composite_loss_weighting"]` is not None,
    `get_alpha_scheduler` must be called with that config and the returned
    object should be stored in `state["composite_loss_weight_schedule"]`.
    Otherwise, the scheduler should be `None` and the registry function should
    not be called.
    """
    results, _ = tmp_pipeline

    # Update config to set/unset the composite loss weighting schedule.
    cfg_path = Path(results) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["training"]["loss"]["params"]["composite_loss_weighting"] = schedule_cfg
    cfg_path.write_text(json.dumps(cfg))

    # Spy on get_alpha_scheduler and return a sentinel.
    calls = {"args": []}
    sentinel = object()

    def spy_get_alpha_scheduler(arg):
        calls["args"].append(arg)
        return sentinel

    monkeypatch.setattr(bt, "get_alpha_scheduler", spy_get_alpha_scheduler)

    # Build components and inspect the returned state.
    trainer = DummyTrainer(mist_args)
    state = trainer.build_components(rank=0, world_size=1)

    if schedule_cfg is None:
        assert state["composite_loss_weighting"] is None
        assert not calls["args"] # Not called when config is None.
    else:
        assert state["composite_loss_weighting"] is sentinel
        assert calls["args"] == [schedule_cfg]  # Called exactly once.
