"""Unit tests for alpha schedulers."""
import pytest

# MIST imports.
from mist.loss_functions import alpha_schedulers


def test_constant_scheduler_returns_fixed_value():
    """Test that the constant scheduler returns a fixed alpha value."""
    scheduler = alpha_schedulers.ConstantScheduler(num_epochs=5, value=0.75)
    for epoch in range(10):
        assert scheduler(epoch) == 0.75


def test_linear_scheduler_during_warmup():
    """Test linear scheduler returns alpha=1.0 during warmup phase."""
    scheduler = alpha_schedulers.LinearScheduler(num_epochs=20, init_pause=3)
    for epoch in range(4):
        assert scheduler(epoch) == 1.0


def test_linear_scheduler_decays_to_zero():
    """Test that linear scheduler decays alpha linearly to 0 after warmup."""
    scheduler = alpha_schedulers.LinearScheduler(num_epochs=10, init_pause=2)

    expected = {
        0: 1.0,                # Warmup.
        1: 1.0,                # Warmup.
        2: 1.0,                # Start of decay.
        3: 0.8571428571428572,
        4: 0.7142857142857143,
        5: 0.5714285714285714,
        6: 0.4285714285714286,
        7: 0.2857142857142857,
        8: 0.1428571428571429,
        9: 0.0,                # End of decay.
    }
    for epoch, expected_alpha in expected.items():
        assert pytest.approx(scheduler(epoch), abs=1e-6) == expected_alpha


def test_get_scheduler_constant():
    """Test factory returns ConstantScheduler."""
    sched = alpha_schedulers.get_alpha_scheduler(
        "constant", value=0.3, num_epochs=10
    )
    assert isinstance(sched, alpha_schedulers.ConstantScheduler)
    assert sched(0) == 0.3


def test_get_scheduler_linear():
    """Test factory returns LinearScheduler."""
    sched = alpha_schedulers.get_alpha_scheduler("linear", num_epochs=10)
    assert isinstance(sched, alpha_schedulers.LinearScheduler)
    assert sched(0) == 1.0


def test_get_scheduler_invalid_name_raises():
    """Test that unknown scheduler name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown scheduler:"):
        alpha_schedulers.get_alpha_scheduler("cosine", num_epochs=10)


def test_list_registered_schedulers():
    """Test that the list of registered schedulers is correct."""
    schedulers = alpha_schedulers.list_alpha_schedulers()
    assert "constant" in schedulers
    assert "linear" in schedulers
    assert len(schedulers) == 2
