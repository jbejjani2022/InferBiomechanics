import os
import hashlib
import shutil
from typing import Dict, List
from cli.abstract_command import AbstractCommand
import argparse


class CreateSplitsCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('create-splits', help='Turn a processed/ folder in train/ and dev/ splits.')
        subparser.add_argument('--data-folder', type=str, help='The folder where the processed/ folder lives.')

    def run(self, args: argparse.Namespace):
        """
        Iterate over all *.b3d files in a directory hierarchy,
        compute file hash, and move to train or dev directories.
        """
        if 'command' in args and args.command != 'create-splits':
            return False

        data_folder: str = os.path.abspath(args.data_folder)

        # Modify these paths as needed
        base_dir = os.path.join(data_folder, 'processed')
        train_dir = os.path.join(data_folder, 'train')
        dev_dir = os.path.join(data_folder, 'dev')

        if not os.path.exists(base_dir):
            print('ERROR: Could not find processed/ folder in ' + data_folder)
            print('No folder at ' + base_dir)

        # Ensure output directories exist
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(dev_dir, exist_ok=True)
        dataset_files: Dict[str, List[str]] = {}
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.b3d'):
                    dataset_name = file_path.split('/')[-3]
                    if dataset_name not in dataset_files:
                        dataset_files[dataset_name] = []
                    dataset_files[dataset_name].append(file_path)

        for dataset_name in dataset_files:
            for i, file in enumerate(sorted(dataset_files[dataset_name])):
                file_name = file.split('/')[-1]
                desired_name = dataset_name + '_' + file_name
                print(desired_name)
                if i > 3:
                    if not os.path.exists(os.path.join(train_dir, desired_name)):
                        shutil.copy(file, os.path.join(train_dir, desired_name))
                else:
                    if not os.path.exists(os.path.join(dev_dir, desired_name)):
                        shutil.copy(file, os.path.join(dev_dir, desired_name))
