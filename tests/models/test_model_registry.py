"""Unit tests for the model registry mechanism in MIST."""
import pytest

# MIST imports.
from mist.models.model_registry import (
    MODEL_REGISTRY,
    register_model,
    get_model_from_registry,
    list_registered_models,
)


@pytest.fixture(autouse=True)
def clear_registry():
    """Ensure model registry is clean before each test."""
    MODEL_REGISTRY.clear()
    yield
    MODEL_REGISTRY.clear()


def test_register_model_success():
    """Test that a model can be successfully registered and retrieved."""
    @register_model("dummy")
    def build_dummy():
        return "dummy_model"

    assert "dummy" in MODEL_REGISTRY
    assert get_model_from_registry("dummy") == "dummy_model"


def test_register_model_duplicate_name():
    """Test that registering a model with a duplicate name raises an error."""
    @register_model("dup")
    def model_one():
        return "one"

    with pytest.raises(ValueError, match="Model 'dup' is already registered"):
        @register_model("dup")
        def model_two():
            return "two"


def test_get_model_from_registry_not_found():
    """Test that requesting an unregistered model raises an error."""
    with pytest.raises(ValueError, match="Model 'missing' is not registered"):
        get_model_from_registry("missing")


def test_list_registered_models_returns_sorted():
    """Test that list_registered_models returns sorted model names."""
    @register_model("z_model")
    def z(): return None

    @register_model("a_model")
    def a(): return None

    registered = list_registered_models()
    assert registered == ["a_model", "z_model"]
