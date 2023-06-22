import json
import os

from analyze_data.analyze import Analyze
from preprocess_data.preprocess import preprocess_dataset
from runtime.args import get_main_args
from runtime.run import RunTime
from runtime.utils import create_empty_dir, set_warning_levels


def create_folders(args):
    data_path = os.path.abspath(args.data)
    with open(data_path, "r") as file:
        data = json.load(file)

    is_test = False
    if "test-data" in data.keys():
        is_test = True

    results = os.path.abspath(args.results)
    processed_data = os.path.abspath(args.processed_data)

    dirs_to_create = [results,
                      os.path.join(results, "predictions"),
                      os.path.join(results, "predictions", "train"),
                      os.path.join(results, "predictions", "train", "raw"),
                      os.path.join(results, "predictions", "train", "postprocess"),
                      os.path.join(results, "predictions", "train", "postprocess", "clean_mask"),
                      os.path.join(results, "predictions", "train", "final"),
                      os.path.join(results, "models"),
                      os.path.join(results, "models", "best"),
                      os.path.join(results, "models", "last")]

    labels = data["labels"]
    for i in range(1, len(labels)):
        dirs_to_create.append(os.path.join(results, "predictions", "train", "postprocess", str(labels[i])))

    if is_test:
        dirs_to_create.append(os.path.join(results, "predictions", "test"))

    for folder in dirs_to_create:
        create_empty_dir(folder)

    create_empty_dir(processed_data)


def main(args):

    set_warning_levels()

    # Create file structure for MIST output
    create_folders(args)

    if args.exec_mode == "all":
        analyze = Analyze(args)
        analyze.run()

        preprocess_dataset(args)

        runtime = RunTime(args)
        runtime.run()

    elif args.exec_mode == "analyze":
        analyze = Analyze(args)
        analyze.run()

    elif args.exec_mode == "preprocess":
        preprocess_dataset(args)

    elif args.exec_mode == "train":
        runtime = RunTime(args)
        runtime.run()


if __name__ == "__main__":
    args = get_main_args()
    main(args)
