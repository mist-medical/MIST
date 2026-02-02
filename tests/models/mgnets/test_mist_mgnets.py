"""Unit tests for the MGNet model implementation."""
import pytest
import torch

# MIST imports.
from mist.models.mgnets.mist_mgnets import MGNet


@pytest.mark.parametrize("mg_type", ["wnet", "fmgnet"])
@pytest.mark.parametrize("depth", [1, 3, 5])
def test_mgnet_forward_eval(mg_type, depth):
    """Test MGNet forward pass in eval mode (inference only)."""
    model = MGNet(
        mg_net=mg_type,
        in_channels=1,
        out_channels=3,
        depth=depth,
        use_residual_blocks=False,
        use_deep_supervision=False,
    )
    model.eval()
    x = torch.randn(1, 1, 64, 64, 64)
    y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == 1
    assert y.shape[1] == 3


@pytest.mark.parametrize("mg_type", ["wnet", "fmgnet"])
@pytest.mark.parametrize("depth", [2, 4])
def test_mgnet_forward_train_without_deep_supervision(mg_type, depth):
    """Test MGNet forward pass in training mode without deep supervision."""
    model = MGNet(
        mg_net=mg_type,
        in_channels=1,
        out_channels=3,
        depth=depth,
        use_residual_blocks=True,
        use_deep_supervision=False,
    )
    model.train()
    x = torch.randn(1, 1, 64, 64, 64)
    output = model(x)
    assert isinstance(output, dict)
    assert "prediction" in output
    assert isinstance(output["prediction"], torch.Tensor)
    assert output.get("deep_supervision") is None


@pytest.mark.parametrize("mg_type", ["wnet", "fmgnet"])
def test_mgnet_forward_train_with_deep_supervision(mg_type):
    """Test MGNet forward pass in training mode with deep supervision."""
    model = MGNet(
        mg_net=mg_type,
        in_channels=1,
        out_channels=2,
        depth=4,
        use_residual_blocks=False,
        use_deep_supervision=True,
        num_deep_supervision_heads=2,
    )
    model.train()
    x = torch.randn(1, 1, 64, 64, 64)
    output = model(x)
    assert isinstance(output, dict)
    assert "prediction" in output
    assert "deep_supervision" in output
    assert isinstance(output["deep_supervision"], tuple)
    for ds in output["deep_supervision"]:
        assert isinstance(ds, torch.Tensor)
        assert ds.shape[0] == 1
        assert ds.shape[1] == 2


def test_invalid_depth_raises():
    """Test that invalid MGNet depth raises ValueError."""
    with pytest.raises(ValueError, match="Depth must be between 1 and 5"):
        MGNet("wnet", in_channels=1, out_channels=2, depth=0)


def test_invalid_mgnet_type_raises():
    """Test that invalid MG architecture name raises ValueError."""
    with pytest.raises(ValueError, match="Invalid MG architecture"):
        MGNet("badnet", in_channels=1, out_channels=2, depth=2)


def test_invalid_supervision_heads_raises():
    """Test that deep supervision heads exceeding depth raises ValueError."""
    with pytest.raises(ValueError, match="Deep supervision heads must be less"):
        MGNet(
            "wnet",
            in_channels=1,
            out_channels=2,
            depth=2,
            use_deep_supervision=True,
            num_deep_supervision_heads=3
        )


def test_get_in_decoder_channels_invalid_architecture():
    """_get_in_decoder_channels raises ValueError for invalid architecture."""
    model = MGNet(
        mg_net="wnet",  # Use valid init so constructor passes.
        in_channels=1,
        out_channels=3,
        depth=2,
    )
    model.mg_net = "invalid"  # Manually override to an invalid string.

    with pytest.raises(ValueError, match="Invalid MG architecture"):
        model._get_in_decoder_channels(2)
