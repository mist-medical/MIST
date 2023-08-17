import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from runtime.args import non_negative_int, float_0_1, ArgParser
from runtime.utils import set_warning_levels, set_memory_growth, set_tf_flags, set_amp, set_xla
from inference.main_inference import check_test_time_input, load_test_time_models, test_time_inference


def get_predict_args():
    p = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.arg("--models", type=str, help="Directory containing saved models")
    p.arg("--config", type=str, help="Path and name of config.json file from results of MIST pipeline")
    p.arg("--data", type=str, help="CSV or JSON file containing paths to data")
    p.arg("--output", type=str, help="Directory to save predictions")
    p.boolean_flag("--fast", default=False, help="Use only one model for prediction to speed up inference time")
    p.arg("--gpu", type=non_negative_int, default=0, help="GPU id to run inference on")
    p.boolean_flag("--amp", default=False, help="Use automatic mixed precision")
    p.boolean_flag("--xla", default=False, help="Use XLA")

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

    args = p.parse_args()
    return args


def main(args):

    # Set warning levels
    set_warning_levels()

    # Horovod init
    hvd_init()

    # Set TF flags
    set_tf_flags()

    # Set visible device to GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Allow memory growth
    set_memory_growth()

    # Set AMP and XLA if required
    if args.amp:
        set_amp()

    if args.xla:
        set_xla()

    # Handle inputs
    df = check_test_time_input(args.data, args.output)

    # Load models
    models = load_test_time_models(args.models, args.fast)

    # Run test time inference
    test_time_inference(df, args.output, args.config, models, args.sw_overlap, args.blend_mode, args.tta)


if __name__ == "__main__":
    args = get_predict_args()
    main(args)
