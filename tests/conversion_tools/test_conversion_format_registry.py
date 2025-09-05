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
"""Tests for mist.conversion_tools.conversion_format_registry."""
from types import SimpleNamespace
import pytest

# MIST imports.
from mist.conversion_tools import conversion_format_registry as reg


def _make_sentinel(name: str):
    """Return a unique callable sentinel with a readable repr."""
    def fn(*args, **kwargs):
        return SimpleNamespace(name=name, args=args, kwargs=kwargs)
    fn.__name__ = f"sentinel_{name}"
    return fn


def test_get_supported_formats_reflects_registry_keys(monkeypatch):
    """get_supported_formats returns the current registry keys."""
    msd_fn = _make_sentinel("msd")
    csv_fn = _make_sentinel("csv")
    monkeypatch.setattr(
        reg, "CONVERSION_REGISTRY", {"msd": msd_fn, "csv": csv_fn}
    )
    formats = reg.get_supported_formats()
    assert set(formats) == {"msd", "csv"}


def test_get_conversion_function_returns_registered_callable(monkeypatch):
    """get_conversion_function returns the exact callable registered."""
    msd_fn = _make_sentinel("msd")
    csv_fn = _make_sentinel("csv")
    monkeypatch.setattr(
        reg, "CONVERSION_REGISTRY", {"msd": msd_fn, "csv": csv_fn}
    )

    got_msd = reg.get_conversion_function("msd")
    got_csv = reg.get_conversion_function("csv")

    assert got_msd is msd_fn
    assert got_csv is csv_fn

    # Sanity: returned objects are callable and behave like our sentinels.
    out = got_msd(1, foo=2)
    assert out.name == "msd" and out.args == (1,) and out.kwargs == {"foo": 2}


def test_get_conversion_function_unknown_raises_keyerror(monkeypatch):
    """Unknown format names should raise KeyError with a clear message."""
    msd_fn = _make_sentinel("msd")
    monkeypatch.setattr(reg, "CONVERSION_REGISTRY", {"msd": msd_fn})

    with pytest.raises(KeyError) as excinfo:
        reg.get_conversion_function("not-a-format")

    msg = str(excinfo.value)
    assert "not-a-format" in msg
    assert "not a registered conversion format" in msg
