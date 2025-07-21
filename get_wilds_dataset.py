# This script downloads the FMoW dataset locally and optionally compress to .tar.gz format

import os
import tarfile
from argparse import ArgumentParser

from wilds import get_dataset, supported_datasets

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=supported_datasets, help="Name of WILDS dataset to download")
    parser.add_argument("--version", type=str, default='1.1', help="Version of the dataset to download")
    parser.add_argument("--compress", action='store_true', help="Compress the dataset to .tar.gz format")
    args = parser.parse_args()

    data_folder = rf"data\{args.dataset}_v{args.version}"
    data_folder_tar = f"{data_folder}.tar.gz"

    if not os.path.exists(data_folder):
        get_dataset(dataset=args.dataset, download=True)

    # Compress to .tar.gz
    if args.compress and not os.path.exists(data_folder_tar):
        tar = tarfile.open(data_folder_tar, "w:gz")
        tar.add(data_folder, arcname=f"{args.dataset}_v{args.version}")
        tar.close()


if __name__ == '__main__':
    main()