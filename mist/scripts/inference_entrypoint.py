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
"""Command line tool MIST inference on a given dataset."""
import os
from argparse import ArgumentDefaultsHelpFormatter
import pandas as pd
import torch

# MIST imports.
from mist.runtime.args import float_0_1, ArgParser
from mist.runtime import utils
from mist.inference import inference_utils
from mist.inference import inference_runners


def get_inference_args():
    """Get command line arguments for MIST prediction."""
    p = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Required parameters.
    p.arg(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing saved models"
    )
    p.arg(
        "--config",
        type=str,
        required=True,
        help="Path and name of config.json file from results of MIST pipeline"
    )
    p.arg(
        "--paths-csv",
        type=str,
        required=True,
        help="CSV file containing paths to images to run prediction on multiple cases"
    )
    p.arg(
        "--output",
        type=str,
        required=True,
        help="Directory to save predictions in NIfTI format"
    )

    # Optional parameters.
    p.boolean_flag(
        "--fast",
        default=False,
        help="Only use first model in ensemble for faster inference"
    )
    p.arg(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on ('cuda', '0', or 'cpu')"
    )
    p.arg(
        "--postprocess-strategy",
        type=str,
        default=None,
        help="Path to postprocessing strategy JSON file"
    )

    # Sliding window parameters.
    p.arg(
        "--sw-overlap",
        type=float_0_1,
        default=0.5,
        help="Amount of overlap between patches during sliding window inference"
    )
    p.arg(
        "--blend-mode",
        type=str,
        choices=["gaussian", "constant"],
        default="gaussian",
        help="How to blend patch predictions from overlapping windows"
    )
    p.boolean_flag("--tta", default=False, help="Use test time augmentation")
    p.boolean_flag(
        "--no-preprocess",
        default=False,
        help="Turn off preprocessing if raw input files are already preprocessed"
    )

    args = p.parse_args()
    return args


def main(args):
    """Main function for MIST inference script."""
    # Set warning levels.
    utils.set_warning_levels()

    # Set device.
    if args.device != "cpu" or args.device != "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = torch.device(int(args.device))
    else:
        device = torch.device(args.device)

    # Validate the input paths CSV file. This should contain an 'id' column
    # and at least one other column pointing to valid NIfTI files.
    if not os.path.exists(args.paths_csv):
        raise FileNotFoundError(
            f"Paths CSV file {args.paths_csv} does not exist."
        )
    dataframe = pd.read_csv(args.paths_csv)
    inference_utils.validate_paths_dataframe(dataframe)

    # Load the MIST configuration.
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} does not exist.")
    mist_configuration = utils.read_json_file(args.config)

    # Run inference.
    with torch.no_grad():
        inference_runners.infer_from_dataframe(
            paths_dataframe=dataframe,
            output_directory=args.output,
            mist_configuration=mist_configuration,
            models_directory=args.models_dir,
            ensemble_models=not args.fast,
            test_time_augmentation=args.tta,
            skip_preprocessing=args.no_preprocess,
            postprocessing_strategy_filepath=args.postprocess_strategy,
            device=device,
        )


def inference_entry():
    """Entry point for MIST inference script."""
    args = get_inference_args()
    main(args)


if __name__ == "__main__":
    inference_entry()
