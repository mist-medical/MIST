"""Entrypoint for the analyze command."""
from mist.main import main
from mist.runtime import utils
from mist.runtime import args


def analyze_entry():
    """Entrypoint for the analyze command."""
    utils.set_warning_levels()
    mist_arguments = args.get_main_args()
    mist_arguments.exec_mode = "analyze"
    main(mist_arguments)


if __name__ == "__main__":
    analyze_entry()
