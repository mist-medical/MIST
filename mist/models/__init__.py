# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Initialize and register all MIST model variants."""

# Trigger registration of all model variants.
from mist.models.nnunet.nnunet_registry import create_nnunet
from mist.models.mednext.mednext_registry import (
    create_mednext_small,
    create_mednext_base,
    create_mednext_medium,
    create_mednext_large,
)
from mist.models.mgnets.mgnets_registry import (
    create_fmgnet,
    create_wnet,
)