import argparse

import torch
from torch.utils.data import DataLoader
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from models.FeedForwardRegressionBaseline import FeedForwardBaseline
from loss.RegressionLossEvaluator import RegressionLossEvaluator
from typing import Any, Dict, Tuple, List
from cli.abstract_command import AbstractCommand
import os
import time
import logging
import numpy as np
import csv

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
        subparser.add_argument('--short', type=bool, default=False, help='Use very short datasets to test without loading a bunch of data.')
        
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
        components = {0: "left-x", 1: "left-y", 2: "left-z", 3: "right-x", 4: "right-y", 5: "right-z"}
        for subject_index in range(0, len(train_dataset.subject_paths)):
            trials = train_dataset.subjects[subject_index].getNumTrials()
            stats: List[Dict[str, Any]] = []
            for trial in range(trials):
                dataset_creation = time.time()
                train_dataset.prepare_data_for_subset([subject_index], [trial])
                # Create a DataLoader to load the data in batches
                if len(train_dataset.windows) == 0:
                    continue
                train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset.windows), shuffle=False)
                dataset_creation = time.time() - dataset_creation
                logging.info(f"Train Subject Index: {subject_index}/{len(train_dataset.subject_paths)}, Trial index: {trial}/{trials} {dataset_creation=}")
            
                loss_evaluator = RegressionLossEvaluator(dataset=train_dataset)
                for i, batch in enumerate(train_dataloader):
                    inputs: Dict[str, torch.Tensor]
                    labels: Dict[str, torch.Tensor]
                    batch_subject_indices: List[int]
                    batch_trial_indices: List[int]
                    inputs, labels, batch_subject_indices, batch_trial_indices = batch

                    # Forward pass
                    outputs = model(inputs, [(train_dataset.skeletons[i], train_dataset.skeletons_contact_bodies[i]) for i in batch_subject_indices])

                    # Compute the loss
                    loss_evaluator(inputs, outputs, labels, batch_subject_indices, batch_trial_indices, compute_report=True, analyze=True, plot_path_root=train_plot_path_root)
                    component_wise_loss_percentiles = np.percentile(loss_evaluator.plot_ferror, 75, axis=0)
                    stats.append({
                        "sub_name": f"{os.path.basename(train_dataset.subject_paths[subject_index])}",
                        "trial_name": f"{train_dataset.subjects[subject_index].getTrialName(trial)}",
                        **{f'force_loss_{components[i]}': component_wise_loss_percentiles[i] for i in range(6)}
                    })
                loss_evaluator.print_report()
        
            with open(os.path.join(checkpoint_dir, 'train_analysis.csv'), 'a') as csvfile: 
                writer = csv.DictWriter(csvfile, fieldnames=stats[0].keys()) 
                writer.writerows(stats)

        # At the end of each epoch, evaluate the model on the dev set
        dev_loss_evaluator = RegressionLossEvaluator(dataset=dev_dataset)
        for subject_index in range(0, len(dev_dataset.subject_paths)):
            trials = dev_dataset.subjects[subject_index].getNumTrials()
            for trial in range(trials):
                dataset_creation = time.time()
                dev_dataset.prepare_data_for_subset([subject_index], [trial])
                dev_dataloader = DataLoader(dev_dataset, batch_size=len(dev_dataset.windows), shuffle=False)
                dataset_creation = time.time() - dataset_creation
                logging.info(f"Dev batch: {subject_index}/{len(dev_dataset.subject_paths)}, Trial index: {trial}/{trials} {dataset_creation=}")
            
                with torch.no_grad():
                    for i, batch in enumerate(dev_dataloader):
                        inputs: Dict[str, torch.Tensor]
                        labels: Dict[str, torch.Tensor]
                        batch_subject_indices: List[int]
                        batch_trial_indices: List[int]
                        inputs, labels, batch_subject_indices, batch_trial_indices = batch
                        outputs = model(inputs, [(dev_dataset.skeletons[i], dev_dataset.skeletons_contact_bodies[i]) for i in batch_subject_indices])
                        dev_loss_evaluator(inputs, outputs, labels, batch_subject_indices, batch_trial_indices, compute_report=True, analyze=True, plot_path_root=dev_plot_path_root)
                dev_loss_evaluator.print_report()
        return True

