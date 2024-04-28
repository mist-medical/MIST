from mist.main import main
from mist.runtime.utils import set_warning_levels
from mist.runtime.args import get_main_args


def analyze_entry():
    set_warning_levels()
    args = get_main_args()
    args.exec_mode = "analyze"
    main(args)


if __name__ == "__main__":
    analyze_entry()
