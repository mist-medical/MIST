"""Main script for MIST."""
import os
import argparse
import numpy as np
import torch

# Import MIST modules.
from mist.analyze_data.analyze import Analyzer
from mist.preprocess_data.preprocess import preprocess_dataset
from mist.runtime import args
from mist.runtime.run import Trainer
from mist.evaluate_preds.evaluate import evaluate
from mist.runtime import utils
from mist.inference import main_inference


def create_folders(arguments: argparse.Namespace) -> None:
    """Create folders for the output of MIST."""
    # Get path to dataset description JSON file.
    data_path = os.path.abspath(arguments.data)

    # Check if test data is present.
    has_test = utils.has_test_data(data_path)

    # Create folders for the output of MIST.
    results = os.path.abspath(arguments.results)
    os.makedirs(results, exist_ok=True)
    os.makedirs(os.path.join(results, "models"), exist_ok=True)
    os.makedirs(
        os.path.join(results, "predictions", "train", "raw"), exist_ok=True
    )

    # Create numpy folder if not in analyze mode. This folder is used to store
    # preprocessed data in numpy format.
    if arguments.exec_mode != "analyze":
        os.makedirs(os.path.abspath(arguments.numpy), exist_ok=True)

    # Create folders for test set predictions if test data is present.
    if has_test:
        os.makedirs(os.path.join(results, "predictions", "test"), exist_ok=True)


def main(arguments: argparse.Namespace) -> None:
    """Main function for MIST."""
    # Create file structure for MIST output.
    create_folders(arguments)

    # Run all or individual pipelines.
    # Analysis pipeline.
    if arguments.exec_mode == "all" or arguments.exec_mode == "analyze":
        analyzer = Analyzer(arguments)
        analyzer.run()

    # Preprocessing pipeline.
    if arguments.exec_mode == "all" or arguments.exec_mode == "preprocess":
        preprocess_dataset(arguments)

    # Training pipeline.
    if arguments.exec_mode == "all" or arguments.exec_mode == "train":
        # Train models for specified folds.
        mist_trainer = Trainer(arguments)
        mist_trainer.fit()

        # Test on each specified fold.
        for fold in arguments.folds:
            main_inference.test_on_fold(arguments, fold)

        # Evaluate predictions from cross-validation.
        evaluate(
            config_json=os.path.join(arguments.results, "config.json"),
            paths_to_predictions=os.path.join(
                arguments.results, "train_paths.csv"
            ),
            source_dir=os.path.join(
                arguments.results, "predictions", "train", "raw"
            ),
            output_csv=os.path.join(arguments.results, "results.csv"),
            list_of_metrics=arguments.metrics,
            use_unit_spacing=arguments.use_unit_spacing,
            surf_dice_tol=arguments.surf_dice_tol,
        )

    # Inference pipeline. Run inference on test set.
    if arguments.exec_mode == "all" or arguments.exec_mode == "train":
        if utils.has_test_data(arguments.data):
            # Get paths to test set files.
            test_df = utils.get_files_df(arguments.data, "test")
            test_df.to_csv(
                os.path.join(arguments.results, "test_paths.csv"), index=False
            )

            # Load models for test-time inference.
            models = main_inference.load_test_time_models(
                os.path.join(arguments.results, "models"), False
            )
            models = [model.eval() for model in models]
            models = [model.to("cuda") for model in models]

            with torch.no_grad():
                main_inference.test_time_inference(
                    df=test_df,
                    dest=os.path.join(arguments.results, "predictions", "test"),
                    config_file=os.path.join(arguments.results, "config.json"),
                    models=models,
                    overlap=arguments.sw_overlap,
                    blend_mode=arguments.blend_mode,
                    tta=arguments.tta,
                    no_preprocess=arguments.no_preprocess,
                    output_std=arguments.output_std,
                )


if __name__ == "__main__":
    # Set warning levels.
    utils.set_warning_levels()

    # Get arguments.
    mist_arguments = args.get_main_args()

    # Set seed.
    utils.set_seed(mist_arguments.seed_val)

    # Check if loss function is compatible with DTM.
    if (
        mist_arguments.loss in ["bl", "hdl", "gsl"] and
        not mist_arguments.use_dtms
    ):
        raise AssertionError(
            f"Loss function {mist_arguments.loss} requires DTM. Use --use_dtm"
        )

    # Check parameters for training.
    if mist_arguments.exec_mode == "all" or mist_arguments.exec_mode == "train":
        # Only overwrite if the user has specified it.
        if not mist_arguments.overwrite:
            if os.path.exists(
                os.path.join(mist_arguments.results, "results.csv")
            ):
                raise AssertionError(
                    "Results folder already contains a previous run. Enable "
                    "--overwrite to overwrite the previous run"
                )

        # Check if the number of folds is compatible with the number of folds
        # specified.
        if (
            np.max(mist_arguments.folds) + 1 < mist_arguments.nfolds or
            len(mist_arguments.folds) > mist_arguments.nfolds
        ):
            raise AssertionError(
                f"More folds listed than specified! Specified "
                f"{mist_arguments.nfolds} folds, but listed the following "
                f"folds {mist_arguments.folds}"
            )

        # Check that number of GPUs is compatible with batch size. Set batch
        # size to be compatible with number of GPUs if not specified.
        n_gpus = utils.set_visible_devices(mist_arguments)

        if mist_arguments.batch_size is None:
            mist_arguments.batch_size = 2 * n_gpus

        if mist_arguments.batch_size % n_gpus != 0:
            raise AssertionError(
                f"Batch size {mist_arguments.batch_size} is not compatible "
                f"number of GPUs {n_gpus}"
            )

    main(mist_arguments)
