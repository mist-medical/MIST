"""Entrypoint for preprocessing pipeline."""
from mist.main import main
from mist.runtime import utils
from mist.runtime import args


def preprocess_entry():
    """Entrypoint for preprocessing pipeline."""
    utils.set_warning_levels()
    mist_arguments = args.get_main_args()
    mist_arguments.exec_mode = "preprocess"
    main(mist_arguments)


if __name__ == "__main__":
    preprocess_entry()
