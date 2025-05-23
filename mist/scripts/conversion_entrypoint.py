# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Command line tool for converting datasets to MIST format.

This script converts datasets from MSD or CSV format to MIST format.
The script takes in the following arguments:
    --format: Format of dataset to be converted (msd or csv)
    --msd-source: Directory containing MSD formatted dataset
    --train-csv: Path to and name of csv containing training ids, mask, and images
    --test-csv: Path to and name of csv containing test ids and images
    --dest: Directory to save converted, MIST formatted dataset
"""
from argparse import ArgumentDefaultsHelpFormatter

# MIST imports.
from mist.conversion_tools.msd import convert_msd
from mist.conversion_tools.csv import convert_csv
from mist.runtime.args import ArgParser


def get_conversion_args():
    """Parse command line arguments for dataset conversion."""
    p = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.arg(
        "--format",
        type=str,
        default="msd",
        choices=["msd", "csv"],
        help="Format of dataset to be converted"
    )
    p.arg(
        "--msd-source",
        type=str,
        help="Directory containing MSD formatted dataset"
    )
    p.arg(
        "--train-csv",
        type=str,
        help="Path to CSV containing patient ids, mask, and images"
    )
    p.arg(
        "--test-csv",
        type=str,
        help="Path to CSV containing test ids and images"
    )
    p.arg(
        "--output",
        type=str,
        help="Directory to save converted MIST formatted dataset"
    )

    args = p.parse_args()
    return args


def main(args):
    if args.format == "msd":
        convert_msd(args.msd_source, args.dest)
    if args.format == "csv":
        convert_csv(args.train_csv, args.dest, args.test_csv)
    raise ValueError(
        f"Invalid format '{args.format}'. Supported formats are 'msd' and "
        "'csv'."
    )


def conversion_entry():
    args = get_conversion_args()
    main(args)


if __name__ == "__main__":
    conversion_entry()
