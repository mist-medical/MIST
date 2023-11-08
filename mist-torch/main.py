import json
import os
import numpy as np

import torch

from analyze_data.analyze import Analyzer
from preprocess_data.preprocess import preprocess_dataset
from runtime.args import get_main_args
from runtime.run import Trainer
from postprocess_preds.postprocess import Postprocessor
from runtime.evaluate import evaluate
from runtime.utils import create_empty_dir, set_seed, set_warning_levels, set_visible_devices, has_test_data, \
    get_files_df
from inference.main_inference import test_time_inference, load_test_time_models


def create_folders(args):
    data_path = os.path.abspath(args.data)
    with open(data_path, "r") as file:
        data = json.load(file)

    has_test = has_test_data(data_path)

    results = os.path.abspath(args.results)
    numpy = os.path.abspath(args.numpy)

    dirs_to_create = [results,
                      os.path.join(results, "models"),
                      os.path.join(results, "predictions"),
                      os.path.join(results, "predictions", "train"),
                      os.path.join(results, "predictions", "train", "raw")]

    if args.postprocess:
        dirs_to_create += [os.path.join(results, "predictions", "train", "postprocess"),
                           os.path.join(results, "predictions", "train", "postprocess", "clean_mask"),
                           os.path.join(results, "predictions", "train", "final")]

        labels = data["labels"]
        for i in range(1, len(labels)):
            dirs_to_create.append(os.path.join(results, "predictions", "train", "postprocess", str(labels[i])))

    if has_test:
        dirs_to_create.append(os.path.join(results, "predictions", "test"))

    for folder in dirs_to_create:
        create_empty_dir(folder)

    create_empty_dir(numpy)


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
                 os.path.join(args.results, "results.csv"))

        if args.postprocess:
            postprocess = Postprocessor(args)
            postprocess.run()

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
                                    args.tta)


if __name__ == "__main__":
    set_warning_levels()
    args = get_main_args()

    if args.exec_mode == "all" or args.exec_mode == "train":
        assert np.max(args.folds) < args.nfolds or len(args.folds) > args.nfolds,  \
            "More folds listed than specified! Make sure folds are zero-indexed"

        n_gpus = set_visible_devices(args)
        assert args.batch_size % n_gpus == 0, \
            "Batch size {} is not compatible with number of GPUs {}".format(args.batch_size, n_gpus)

    set_seed(args.seed)
    main(args)
