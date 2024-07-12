import argparse

import torch
from torch.utils.data import DataLoader
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from src.loss.dynamics.RegressionLossEvaluator import RegressionLossEvaluator
from typing import Dict, Tuple, List
from cli.abstract_command import AbstractCommand
import os
import time
import wandb
import numpy as np
import logging
import subprocess


class PickleDataCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('pickle-data', help='Preload the data, and write pickled versions to disk.')
        subparser.add_argument('--dataset-home', type=str, default='../data',
                               help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--history-len', type=int, default=50,
                               help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--stride', type=int, default=5,
                               help='The number of timesteps of context to show when constructing the inputs.')

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'pickle-data':
            return False
        dataset_home: str = args.dataset_home
        history_len: int = args.history_len
        stride: int = args.stride

        # Create an instance of the dataset
        DEV = 'test'
        train_dataset_path = os.path.abspath(os.path.join(dataset_home, 'train'))
        train_output_folder: str = os.path.abspath(os.path.join(dataset_home, 'train_pickled'))
        dev_dataset_path = os.path.abspath(os.path.join(dataset_home, DEV))
        dev_output_folder: str = os.path.abspath(os.path.join(dataset_home, 'dev_pickled'))
        logging.info('## Loading training dataset without skeleton:')
        train_dataset = AddBiomechanicsDataset(
                                               train_dataset_path,
                                               history_len,
                                               stride=stride,
                                               device=torch.device('cpu'),
                                               geometry_folder=None,
                                               skip_loading_skeletons=True,
                                               testing_with_short_dataset=False)

        file_num_blocks = 100000

        for i in range(0, len(train_dataset), file_num_blocks):
            print('Saving train dataset block ' + str(i) + ' of ' + str(len(train_dataset)))
            to_save = []
            end_index = min(i + file_num_blocks, len(train_dataset))
            for j in range(i, end_index):
                if j % 1000 == 0:
                    print('Loading train dataset ' + str(j) + ' of ' + str(len(train_dataset)))
                to_save.append(train_dataset[j])
            save_path = os.path.join(train_output_folder, f'train_{i}.pkl')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(to_save, save_path)

        logging.info('## Loading dev dataset without skeleton:')
        dev_dataset = AddBiomechanicsDataset(
                                             dev_dataset_path,
                                             history_len,
                                             stride=stride,
                                             device=torch.device('cpu'),
                                             geometry_folder=None,
                                             skip_loading_skeletons=True,
                                             testing_with_short_dataset=False)

        for i in range(0, len(dev_dataset), file_num_blocks):
            to_save = dev_dataset[i:i+file_num_blocks]
            save_path = os.path.join(dev_output_folder, f'dev_{i}.pkl')
            torch.save(to_save, save_path)

        return True

# python3 main.py train --model feedforward --checkpoint-dir "../checkpoints/checkpoint-gait-ly-only" --prefetch-chunk-size 5 --hidden-dims 32 32 --batchnorm True --dropout True --dropout-prob 0.5 --activation tanh --learning-rate 0.01 --opt-type adagrad --dataset-home "../data" --epochs 500
