import numpy as np

from mist.main import main
from mist.runtime.utils import set_seed, set_warning_levels, set_visible_devices
from mist.runtime.args import get_main_args


def train_entry():
    set_warning_levels()
    args = get_main_args()
    args.exec_mode = "train"

    if args.loss in ["bl", "hdl", "gsl"]:
        args.use_dtm = True

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


if __name__ == "__main__":
    train_entry()
