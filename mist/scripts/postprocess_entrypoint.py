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
"""Command line tool for postprocessing predictions from MIST output."""
from argparse import ArgumentDefaultsHelpFormatter

from mist.postprocessing.postprocessor import Postprocessor
from mist.runtime.args import ArgParser
from mist.runtime import utils


def get_postprocess_args():
    """Parse command line arguments for postprocessing predictions."""
    p = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Required arguments.
    p.arg(
        "--base-predictions",
        type=str,
        required=True,
        help="Path to folder containing base predictions"
    )
    p.arg(
        "--output",
        type=str,
        required=True,
        help="Path to save postprocessed masks"
    )
    p.arg(
        "--postprocess-strategy",
        type=str,
        required=True,
        help="Path to JSON file specifying the postprocessing strategy"
    )

    args = p.parse_args()
    return args


def main(args):
    """Main function for postprocessing predictions."""
    # Set warning levels
    utils.set_warning_levels()

    # Create the postprocessor object and run it.
    postprocessor = Postprocessor(
        strategy_path=args.postprocess_strategy,
    )
    postprocessor.run(
        base_dir=args.base_predictions,
        output_dir=args.output
    )


def postprocess_entry():
    """Entry point for the MIST postprocessing script."""
    args = get_postprocess_args()
    main(args)


if __name__ == "__main__":
    postprocess_entry()
