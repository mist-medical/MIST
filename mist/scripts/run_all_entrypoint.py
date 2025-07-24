"""Entrypoint for running all pipelines in MIST."""
import os
import numpy as np

from mist.main import main
from mist.runtime import utils
from mist.runtime import args


def run_all_entry():
    """Run all pipelines in MIST."""
    # Set warning levels.
    utils.set_warning_levels()

    # Get arguments.
    mist_arguments = args.get_main_args()

    # Set execution mode to all.
    mist_arguments.exec_mode = "all"

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
        np.max(mist_arguments.folds) + 1 > mist_arguments.nfolds or
        len(mist_arguments.folds) > mist_arguments.nfolds
    ):
        raise AssertionError(
            f"More folds listed than specified! Specified "
            f"{mist_arguments.nfolds} folds, but listed the following "
            f"folds {mist_arguments.folds}"
        )

    # Check that number of GPUs is compatible with batch size. Set batch size to
    # be compatible with number of GPUs if not specified.
    n_gpus = utils.set_visible_devices(mist_arguments)

    if mist_arguments.batch_size is None:
        mist_arguments.batch_size = 2 * n_gpus

    if mist_arguments.batch_size % n_gpus != 0:
        raise AssertionError(
            f"Batch size {mist_arguments.batch_size} is not compatible "
            f"number of GPUs {n_gpus}"
        )

    main(mist_arguments)


if __name__ == "__main__":
    run_all_entry()
