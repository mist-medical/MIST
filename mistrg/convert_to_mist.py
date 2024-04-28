import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from mist.conversion_tools.msd import convert_msd
from mist.conversion_tools.csv import convert_csv


class ArgParser(ArgumentParser):
    def arg(self, *args, **kwargs):
        return super().add_argument(*args, **kwargs)


def get_convert_args():
    p = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.arg("--format", type=str, default="msd", choices=["msd", "csv"], help="Format of dataset to be converted")
    p.arg("--msd-source", type=str, help="Directory containing MSD formatted dataset")
    p.arg("--train-csv", type=str, help="Path to and name of csv containing training ids, mask, and images")
    p.arg("--test-csv", type=str, help="Path to and name of csv containing test ids and images")

    p.arg("--dest", type=str, help="Directory to save converted, MIST formatted dataset")

    args = p.parse_args()
    return args


def main(args):
    if args.format == "msd":
        convert_msd(args.msd_source, args.dest)
    elif args.format == "csv":
        convert_csv(args.train_csv, args.test_csv, args.dest)
    else:
        print("Enter valid format type!")


def convert_to_mist_entry():
    args = get_convert_args()
    main(args)


if __name__ == "__main__":
    convert_to_mist_entry()
