import argparse

import torch
from torch.utils.data import DataLoader
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from models.FeedForwardRegressionBaseline import FeedForwardBaseline
from loss.RegressionLossEvaluator import RegressionLossEvaluator
from typing import Dict, Tuple, List
from cli.abstract_command import AbstractCommand
import os
import time
import logging
import numpy as np

class AnalyzeCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('analyze', help='Evaluate the performance of a model on dataset.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--model-type', type=str, default='feedforward', help='The model to train.')
        subparser.add_argument('--checkpoint-dir', type=str, default='../checkpoints', help='The path to a model checkpoint to save during training. Also, starts from the latest checkpoint in this directory.')
        subparser.add_argument('--geometry-folder', type=str, default=None, help='Path to the Geometry folder with bone mesh data.')
        subparser.add_argument('--history-len', type=int, default=5, help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--hidden-size', type=int, default=512, help='The hidden size to use when constructing the model.')
        subparser.add_argument('--device', type=str, default='cpu', help='Where to run the code, either cpu or gpu.')
        subparser.add_argument('--batch-size', type=int, default=32, help='The batch size to use when evaluating the model.')
        subparser.add_argument('--short', type=bool, default=False, help='Use very short datasets to test without loading a bunch of data.')
        subparser.add_argument('--num-subjects-prefetch', type=int, default=1, help='Number of subjects to fetch all the data for in each iteration.')

    def run(self, args: argparse.Namespace):
        """
        Iterate over all *.b3d files in a directory hierarchy,
        compute file hash, and move to train or dev directories.
        """
        if 'command' in args and args.command != 'analyze':
            return False
        model_type: str = args.model_type
        checkpoint_dir: str = os.path.join(os.path.abspath(args.checkpoint_dir), args.model_type)
        history_len: int = args.history_len
        hidden_size: int = args.hidden_size
        batch_size: int = args.batch_size
        device: str = args.device
        short: bool = args.short
        
        train_plot_path_root: str = os.path.join(checkpoint_dir, "analysis/plots/train")
        dev_plot_path_root: str = os.path.join(checkpoint_dir, "analysis/plots/dev")
        if not os.path.isdir(train_plot_path_root):
            os.makedirs(train_plot_path_root)
        if not os.path.isdir(dev_plot_path_root):
            os.makedirs(dev_plot_path_root)

        geometry = self.ensure_geometry(args.geometry_folder)

        # Create an instance of the dataset
        train_dataset_path = os.path.abspath(os.path.join(args.dataset_home, 'train'))
        dev_dataset_path = os.path.abspath(os.path.join(args.dataset_home, 'dev'))
        logging.info('## Loading datasets with skeletons:')
        train_dataset = AddBiomechanicsDataset(train_dataset_path, history_len, device=torch.device(device), geometry_folder=geometry, testing_with_short_dataset=short)
        dev_dataset = AddBiomechanicsDataset(dev_dataset_path, history_len, device=torch.device(device), geometry_folder=geometry, testing_with_short_dataset=short)

        # Create an instance of the model
        model = self.get_model(train_dataset.num_dofs, train_dataset.num_joints, model_type, history_len, hidden_size, device, checkpoint_dir=checkpoint_dir)

        # Iterate over the entire training dataset
        permuted_indices = np.arange(len(train_dataset.subject_paths))
            
        # Iterate over the entire training dataset
        if args.num_subjects_prefetch < 0:
            args.num_subjects_prefetch = len(train_dataset.subject_paths)

        for subject_index in range(0, len(train_dataset.subject_paths), args.num_subjects_prefetch):
            dataset_creation = time.time()
            subset_indices = permuted_indices[subject_index:subject_index+args.num_subjects_prefetch]
            train_dataset.prepare_data_for_subset(subset_indices)
            # Create a DataLoader to load the data in batches
            train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset.windows), shuffle=False)
            dataset_creation = time.time() - dataset_creation
            logging.info(f"Train Subject Index: {subject_index}/{len(train_dataset.subject_paths)} {dataset_creation=}")
        
            loss_evaluator = RegressionLossEvaluator(dataset=train_dataset)
            for i, batch in enumerate(train_dataloader):
                inputs: Dict[str, torch.Tensor]
                labels: Dict[str, torch.Tensor]
                batch_subject_indices: List[int]
                inputs, labels, batch_subject_indices = batch

                # Forward pass
                outputs = model(inputs, [(train_dataset.skeletons[i], train_dataset.skeletons_contact_bodies[i]) for i in batch_subject_indices])

                # Compute the loss
                loss_evaluator(inputs, outputs, labels, batch_subject_indices, compute_report=True, analyze=True, plot_path_root=train_plot_path_root)

            loss_evaluator.print_report(reset=False)

        # Report training loss on this epoch
        print('Training Set Evaluation: ')
        loss_evaluator.print_report()

        # At the end of each epoch, evaluate the model on the dev set
        dev_loss_evaluator = RegressionLossEvaluator(dataset=dev_dataset)
        permuted_indices = np.arange(len(dev_dataset.subject_paths))
        for subject_index in range(0, len(dev_dataset.subject_paths), args.num_subjects_prefetch):
            dataset_creation = time.time()
            subset_indices = permuted_indices[subject_index:subject_index+args.num_subjects_prefetch]
            dev_dataset.prepare_data_for_subset(subset_indices)
            dev_dataloader = DataLoader(dev_dataset, batch_size=len(dev_dataset.windows), shuffle=False)
            dataset_creation = time.time() - dataset_creation
            logging.info(f"Dev batch: {subject_index}/{len(dev_dataset.subject_paths)} {dataset_creation=}")
        
        with torch.no_grad():
            for i, batch in enumerate(dev_dataloader):
                inputs: Dict[str, torch.Tensor]
                labels: Dict[str, torch.Tensor]
                batch_subject_indices: List[int]
                inputs, labels, batch_subject_indices = batch
                outputs = model(inputs, [(dev_dataset.skeletons[i], dev_dataset.skeletons_contact_bodies[i]) for i in batch_subject_indices])
                dev_loss_evaluator(inputs, outputs, labels, batch_subject_indices, compute_report=True, analyze=True, plot_path_root=dev_plot_path_root)
                if i % 100 == 0:
                    print('  - Dev Batch ' + str(i) + '/' + str(len(dev_dataloader)))
                if i % 1000 == 0:
                    dev_loss_evaluator.print_report(reset=False)
        # Report dev loss on this epoch
        print('Dev Set Evaluation: ')
        dev_loss_evaluator.print_report()
        return True

