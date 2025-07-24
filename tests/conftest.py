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
"""Configuration for pytest."""
import sys
import types
from unittest import mock

class DummyPipeline:
    def __init__(self, *args, **kwargs):
        pass

# Create the fake dali structure.
fake_dali = types.SimpleNamespace(
    fn=mock.MagicMock(),
    types=mock.MagicMock(),
    tensors=mock.MagicMock(),
    math=mock.MagicMock(),
    ops=mock.MagicMock(),
    pipeline=types.SimpleNamespace(Pipeline=DummyPipeline),
    plugin=types.SimpleNamespace(pytorch=mock.MagicMock())
)

# Register modules.
sys.modules["nvidia"] = types.SimpleNamespace(dali=fake_dali)
sys.modules["nvidia.dali"] = fake_dali
sys.modules["nvidia.dali.fn"] = fake_dali.fn
sys.modules["nvidia.dali.types"] = fake_dali.types
sys.modules["nvidia.dali.tensors"] = fake_dali.tensors
sys.modules["nvidia.dali.math"] = fake_dali.math
sys.modules["nvidia.dali.ops"] = fake_dali.ops
sys.modules["nvidia.dali.pipeline"] = fake_dali.pipeline
sys.modules["nvidia.dali.plugin"] = fake_dali.plugin
sys.modules["nvidia.dali.plugin.pytorch"] = fake_dali.plugin.pytorch
