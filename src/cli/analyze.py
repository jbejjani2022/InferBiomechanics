import argparse

import torch
from torch.utils.data import DataLoader
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from models.FeedForwardRegressionBaseline import FeedForwardBaseline
from loss.RegressionLossEvaluator import RegressionLossEvaluator
from typing import Any, Dict, Tuple, List
from cli.abstract_command import AbstractCommand
from cli.utilities import get_git_hash, has_uncommitted_changes
import os
import time
import logging
import numpy as np
import csv
import wandb


class AnalyzeCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('analyze', help='Evaluate the performance of a model on dataset.')
        subparser.add_argument('--dataset-home', type=str, default='../data', help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--model-type', type=str, default='feedforward', help='The model to train.')
        subparser.add_argument('--no-wandb', action='store_true', default=False,
                               help='Log this analysis to Weights and Biases.')
        subparser.add_argument('--output-data-format', type=str, default='all_frames', choices=['all_frames', 'last_frame'], 
                               help='Output for all frames in a window or only the last frame.')
        subparser.add_argument('--checkpoint-dir', type=str, default='../checkpoints', help='The path to a model checkpoint to save during training. Also, starts from the latest checkpoint in this directory.')
        subparser.add_argument('--geometry-folder', type=str, default=None, help='Path to the Geometry folder with bone mesh data.')
        subparser.add_argument('--history-len', type=int, default=50,
                               help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--stride', type=int, default=5,
                               help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 512],
                               help='Hidden dims across different layers.')
        subparser.add_argument('--activation', type=str, default='sigmoid', help='Which activation func?')
        subparser.add_argument('--device', type=str, default='cpu', help='Where to run the code, either cpu or gpu.')
        subparser.add_argument('--short', type=bool, default=False, help='Use very short datasets to test without loading a bunch of data.')
        subparser.add_argument('--data-loading-workers', type=int, default=3,
                               help='Number of separate processes to spawn to load data in parallel.')
        subparser.add_argument('--predict-grf-components', type=int, nargs='+', default=[1], help='Which grf components to train.')
        subparser.add_argument('--predict-cop-components', type=int, nargs='+', default=[], help='Which grf components to train.')
        subparser.add_argument('--predict-moment-components', type=int, nargs='+', default=[], help='Which grf components to train.')
        subparser.add_argument('--predict-wrench-components', type=int, nargs='+', default=[], help='Which grf components to train.')

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
        hidden_dims: List[int] = args.hidden_dims
        device: str = args.device
        short: bool = args.short
        data_loading_workers: int = args.data_loading_workers
        stride: int = args.stride
        root_history_len: int = 10
        activation: str = args.activation
        output_data_format: str = args.output_data_format
        log_to_wandb: bool = not args.no_wandb

        train_plot_path_root: str = os.path.join(checkpoint_dir, "analysis/plots/train")
        dev_plot_path_root: str = os.path.join(checkpoint_dir, "analysis/plots/dev")
        if not os.path.isdir(train_plot_path_root):
            os.makedirs(train_plot_path_root)
        if not os.path.isdir(dev_plot_path_root):
            os.makedirs(dev_plot_path_root)

        geometry = self.ensure_geometry(args.geometry_folder)

        # Create an instance of the dataset
        DEV = 'test'
        train_dataset_path = os.path.abspath(os.path.join(args.dataset_home, 'train'))
        dev_dataset_path = os.path.abspath(os.path.join(args.dataset_home, DEV))
        logging.info('## Loading dev dataset with skeletons:')
        dev_dataset = AddBiomechanicsDataset(dev_dataset_path,
                                             history_len,
                                             stride=stride,
                                             device=torch.device(device),
                                             geometry_folder=geometry,
                                             output_data_format=output_data_format,
                                             testing_with_short_dataset=short)

        # Create an instance of the model
        model = self.get_model(dev_dataset.num_dofs,
                               dev_dataset.num_joints,
                               model_type,
                               history_len=history_len,
                               hidden_dims=hidden_dims,
                               activation=activation,
                               stride=stride,
                               batchnorm=False,
                               dropout=False,
                               dropout_prob=0.0,
                               root_history_len=root_history_len,
                               output_data_format=output_data_format,
                               device=device)

        if model_type != 'analytical':
            self.load_latest_checkpoint(model, checkpoint_dir=checkpoint_dir)

        components = {0: "left-x", 1: "left-y", 2: "left-z", 3: "right-x", 4: "right-y", 5: "right-z"}

        # At the end of each epoch, evaluate the model on the dev set
        dev_loss_evaluator = RegressionLossEvaluator(dataset=dev_dataset, split=DEV)
        dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=data_loading_workers)

        has_uncommitted = has_uncommitted_changes()
        if has_uncommitted:
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error(
                "ERROR: UNCOMMITTED CHANGES IN REPO! THIS WILL MAKE IT HARD TO REPLICATE THIS EXPERIMENT LATER")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if log_to_wandb:
            # Grab all cmd args and add current git hash
            config = args.__dict__
            config["git_hash"] = get_git_hash

            logging.info('Initializing wandb...')
            wandb.init(
                # set the wandb project where this run will be logged
                project="addbiomechanics-baseline",

                # track hyperparameters and run metadata
                config=config
            )

        with torch.no_grad():
            for i, batch in enumerate(dev_dataloader):
                inputs: Dict[str, torch.Tensor]
                labels: Dict[str, torch.Tensor]
                batch_subject_indices: List[int]
                batch_trial_indices: List[int]
                inputs, labels, batch_subject_indices, batch_trial_indices = batch

                if model_type == 'analytical':
                    skels_and_contact = [
                        (dev_dataset.skeletons[subject_index], dev_dataset.skeletons_contact_bodies[subject_index])
                        for subject_index in batch_subject_indices
                    ]
                    outputs = model(inputs, skels_and_contact)
                else:
                    outputs = model(inputs)

                dev_loss_evaluator(inputs,
                                   outputs,
                                   labels,
                                   batch_subject_indices,
                                   batch_trial_indices,
                                   args,
                                   compute_report=True) # analyze=True, plot_path_root=dev_plot_path_root
                # component_wise_loss_percentiles = np.percentile(dev_loss_evaluator.plot_ferror, 75, axis=0)
                subject_index = batch_subject_indices[0]
                trial_index = batch_trial_indices[0]
                subject_path: str = dev_dataset.subject_paths[subject_index]
                stats = {
                    "sub_name": f"{os.path.basename(subject_path)}",
                    "trial_name": f"{dev_dataset.subjects[batch_subject_indices[0]].getTrialName(trial_index)}",
                    # **{f'force_loss_{components[i]}': component_wise_loss_percentiles[i] for i in args.predict_grf_components}
                }
                with open(os.path.join(checkpoint_dir, 'dev_analysis.csv'), 'a') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
                    writer.writerow(stats)
                if (i + 1) % 100 == 0 or i == len(dev_dataloader) - 1:
                    logging.info('  - Batch ' + str(i + 1) + '/' + str(len(dev_dataloader)))
                if (i + 1) % 1000 == 0 or i == len(dev_dataloader) - 1:
                    dev_loss_evaluator.print_report(args, reset=False, log_to_wandb=log_to_wandb)
        print('Final dev results:')
        dev_loss_evaluator.print_report(log_to_wandb=log_to_wandb)

        logging.info('## Loading dev dataset with skeletons:')
        train_dataset = AddBiomechanicsDataset(train_dataset_path,
                                               history_len,
                                               stride=stride,
                                               device=torch.device(device),
                                               geometry_folder=geometry,
                                               output_data_format=output_data_format,
                                               testing_with_short_dataset=short)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=data_loading_workers)

        loss_evaluator = RegressionLossEvaluator(dataset=train_dataset,
                                                 split='train')
        for i, batch in enumerate(train_dataloader):
            inputs: Dict[str, torch.Tensor]
            labels: Dict[str, torch.Tensor]
            batch_subject_indices: List[int]
            batch_trial_indices: List[int]
            inputs, labels, batch_subject_indices, batch_trial_indices = batch

            subject_index = batch_subject_indices[0]
            trial_index = batch_trial_indices[0]
            subject_path: str = train_dataset.subject_paths[subject_index]

            # Forward pass
            if model_type == 'analytical':
                skels_and_contact = [
                    (train_dataset.skeletons[subject_index], train_dataset.skeletons_contact_bodies[subject_index])
                    for subject_index in batch_subject_indices
                ]
                outputs = model(inputs, skels_and_contact)
            else:
                outputs = model(inputs)

            # Compute the loss
            loss_evaluator(inputs,
                           outputs,
                           labels,
                           batch_subject_indices,
                           batch_trial_indices,
                           args,
                           compute_report=True) # analyze=True, plot_path_root=train_plot_path_root
            # component_wise_loss_percentiles = np.percentile(loss_evaluator.plot_ferror, 75, axis=0)
            stats = {
                "sub_name": f"{os.path.basename(subject_path)}",
                "trial_name": f"{train_dataset.subjects[batch_subject_indices[0]].getTrialName(trial_index)}",
                # **{f'force_loss_{components[i]}': component_wise_loss_percentiles[i] for i in args.predict_grf_components}
            }
            with open(os.path.join(checkpoint_dir, 'train_analysis.csv'), 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
                writer.writerow(stats)
            if (i + 1) % 100 == 0 or i == len(train_dataloader) - 1:
                logging.info('  - Batch ' + str(i + 1) + '/' + str(len(train_dataloader)))
            if (i + 1) % 1000 == 0 or i == len(train_dataloader) - 1:
                loss_evaluator.print_report(args, reset=False, log_to_wandb=log_to_wandb)
        print('Final train results:')
        loss_evaluator.print_report(log_to_wandb=log_to_wandb)

        return True

# python3 main.py analyze --model analytical --checkpoint-dir "../checkpoints/checkpoint-gait-ly-only" --dataset-home "/n/holyscratch01/pslade_lab/cbrownpinilla/paper_reimplementation/data/addb_dataset"