import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch

from mist.runtime.args import non_negative_int, float_0_1, ArgParser
from mist.runtime.utils import set_warning_levels

from mist.inference.main_inference import (
    check_test_time_input,
    load_test_time_models,
    test_time_inference
)


def get_predict_args():
    p = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.arg("--models", type=str, help="Directory containing saved models")
    p.arg("--config", type=str, help="Path and name of config.json file from results of MIST pipeline")
    p.arg("--data", type=str, help="CSV or JSON file containing paths to images to run prediction on multiple cases")
    p.arg("--output", type=str, help="Directory or path to nifti file to save predictions")
    p.boolean_flag("--fast", default=False, help="Use only one model for prediction to speed up inference time")
    p.arg("--gpu", type=non_negative_int, default=0, help="GPU id to run inference on")

    p.arg("--sw-overlap",
          type=float_0_1,
          default=0.5,
          help="Amount of overlap between scans during sliding window inference")
    p.arg("--blend-mode",
          type=str,
          choices=["gaussian", "constant"],
          default="gaussian",
          help="How to blend output of overlapping windows")
    p.boolean_flag("--tta", default=False, help="Use test time augmentation")
    p.boolean_flag("--no-preprocess", default=False, help="Turn off preprocessing")
    p.boolean_flag("--output-std", default=False, help="Outputs standard deviation image")

    args = p.parse_args()
    return args


def main(args):
    # Set warning levels
    set_warning_levels()

    # Set visible device to GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Handle inputs
    df = check_test_time_input(args.data)

    # Load models
    models = load_test_time_models(os.path.join(args.models), args.fast)
    models = [model.eval() for model in models]
    models = [model.to("cuda") for model in models]

    with torch.no_grad():
        test_time_inference(df,
                            args.output,
                            args.config,
                            models,
                            args.sw_overlap,
                            args.blend_mode,
                            args.tta,
                            args.no_preprocess,
                            args.output_std)


def mist_predict_entry():
    args = get_predict_args()
    main(args)


if __name__ == "__main__":
    mist_predict_entry()
