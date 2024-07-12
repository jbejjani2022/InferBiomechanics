import argparse

import torch
from torch.utils.data import DataLoader
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from models.FeedForwardRegressionBaseline import FeedForwardBaseline
from src.loss.dynamics.RegressionLossEvaluator import RegressionLossEvaluator
from typing import Dict, Tuple, List
from cli.abstract_command import AbstractCommand
import os
import time
import wandb


class SanityCheckCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('sanity-check', help='Sanity check the AddBiomechanics dataset, to ensure it does not contain any unrealistic values.')
        subparser.add_argument('--short', type=bool, default=False, help='Use very short datasets to test without loading a bunch of data.')

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'sanity-check':
            return False
        short: bool = args.short
        with torch.no_grad():
            # Create an instance of the dataset
            print('## Loading TRAIN set:')
            train_dataset = AddBiomechanicsDataset(
                os.path.abspath('../data/train'),
                window_size=1,
                device=torch.device('cpu'),
                geometry_folder=None,
                testing_with_short_dataset=short,
                skip_loading_skeletons=True)
            print('Computing bounds, mean, and variance on inputs and outputs')
            # Dictionary to hold stats for each key in inputs and outputs
            stats = {}

            for i in range(len(train_dataset)):
                inputs: Dict[str, torch.Tensor]
                labels: Dict[str, torch.Tensor]
                batch_subject_index: int
                inputs, labels, batch_subject_index = train_dataset[i]

                if i % 1000 == 0:
                    print(f'Processed {i}/{len(train_dataset)} samples')
                if i % 100000 == 0:
                    print('Intermediate Results:')
                    for key, values in stats.items():
                        print(
                            f"{key} - Mean: {values['mean'] / stats[key]['n']}, Variance: {values['var'] / stats[key]['n']}, Min: {values['min']}, Max: {values['max']}")

                # Function to update stats for a given tensor and key
                def update_stats(key, tensor, is_input=True):
                    if key not in stats:
                        stats[key] = {'mean': 0, 'var': 0, 'min': float('inf'), 'max': float('-inf'), 'n': 0}

                    stats[key]['mean'] += tensor.mean().item()
                    stats[key]['var'] += tensor.var().item()
                    stats[key]['min'] = min(stats[key]['min'], tensor.min().item())
                    stats[key]['max'] = max(stats[key]['max'], tensor.max().item())
                    stats[key]['n'] += 1

                # Updating stats for inputs
                for key, tensor in inputs.items():
                    update_stats(f"input_{key}", tensor)

                # Updating stats for outputs(labels)
                for key, tensor in labels.items():
                    update_stats(f"output_{key}", tensor, False)

            print('Results:')
            for key, values in stats.items():
                print(
                    f"{key} - Mean: {values['mean'] / stats[key]['n']}, Variance: {values['var'] / stats[key]['n']}, Min: {values['min']}, Max: {values['max']}")
