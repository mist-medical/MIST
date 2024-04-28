import os
import numpy as np

import torch

from mist.analyze_data.analyze import Analyzer
from mist.preprocess_data.preprocess import preprocess_dataset
from mist.runtime.args import get_main_args
from mist.runtime.run import Trainer
from mist.evaluate_preds.evaluate import evaluate

from mist.runtime.utils import (
    create_empty_dir,
    set_seed,
    set_warning_levels,
    set_visible_devices,
    has_test_data,
    get_files_df
)

from mist.inference.main_inference import (
    test_time_inference,
    load_test_time_models
)


def create_folders(args):
    data_path = os.path.abspath(args.data)
    has_test = has_test_data(data_path)

    results = os.path.abspath(args.results)
    dirs_to_create = [results,
                      os.path.join(results, "models"),
                      os.path.join(results, "predictions"),
                      os.path.join(results, "predictions", "train"),
                      os.path.join(results, "predictions", "train", "raw")]

    if args.exec_mode != "analyze":
        dirs_to_create.append(os.path.abspath(args.numpy))

    if has_test:
        dirs_to_create.append(os.path.join(results, "predictions", "test"))

    for folder in dirs_to_create:
        create_empty_dir(folder)


def main(args):
    # Create file structure for MIST output
    create_folders(args)
    if args.exec_mode == "all" or args.exec_mode == "analyze":
        analyze = Analyzer(args)
        analyze.run()

    if args.exec_mode == "all" or args.exec_mode == "preprocess":
        preprocess_dataset(args)

    if args.exec_mode == "all" or args.exec_mode == "train":
        mist_trainer = Trainer(args)
        mist_trainer.fit()

        evaluate(args.data,
                 os.path.join(args.results, "train_paths.csv"),
                 os.path.join(args.results, "predictions", "train", "raw"),
                 os.path.join(args.results, "results.csv"),
                 args.metrics,
                 args.use_native_spacing)

    if args.exec_mode == "all" or args.exec_mode == "train":
        if has_test_data(args.data):
            test_df = get_files_df(args.data, "test")
            test_df.to_csv(os.path.join(args.results, "test_paths.csv"), index=False)

            models = load_test_time_models(os.path.join(args.results, "models"), False)
            models = [model.eval() for model in models]
            models = [model.to("cuda") for model in models]

            with torch.no_grad():
                test_time_inference(test_df,
                                    os.path.join(args.results, "predictions", "test"),
                                    os.path.join(args.results, "config.json"),
                                    models,
                                    args.sw_overlap,
                                    args.blend_mode,
                                    args.tta,
                                    args.no_preprocess,
                                    args.output_std)


if __name__ == "__main__":
    set_warning_levels()
    args = get_main_args()

    if args.loss in ["bl", "hdl", "gsl"]:
        args.use_dtm = True

    if args.exec_mode == "all" or args.exec_mode == "train":
        assert np.max(args.folds) < args.nfolds or len(args.folds) > args.nfolds, \
            "More folds listed than specified! Make sure folds are zero-indexed"

        n_gpus = set_visible_devices(args)

        if args.batch_size is None:
            args.batch_size = 2 * n_gpus
        else:
            assert args.batch_size % n_gpus == 0, \
                "Batch size {} is not compatible with number of GPUs {}".format(args.batch_size, n_gpus)

    set_seed(args.seed_val)
    main(args)
