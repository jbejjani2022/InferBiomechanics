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


class TrainCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('train', help='Train a model on the AddBiomechanics dataset')
        subparser.add_argument('--model-type', type=str, default='feedforward', help='The model to train. Currently only supports feedforward.')
        subparser.add_argument('--checkpoint-path', type=str, default='../checkpoints', help='The path to a model checkpoint to save during training. Also, starts from the latest checkpoint in this directory.')
        subparser.add_argument('--geometry-folder', type=str, default=None, help='Path to the Geometry folder with bone mesh data.')
        subparser.add_argument('--history-len', type=int, default=5, help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--hidden-size', type=int, default=512, help='The hidden size to use when constructing the model.')
        subparser.add_argument('--batch-size', type=int, default=512, help='The batch size to use when training the model.')
        subparser.add_argument('--device', type=str, default='cpu', help='Where to run the code, either cpu or gpu.')

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'train':
            return False
        model_type: str = args.model_type
        checkpoint_path: str = os.path.abspath(args.checkpoint_path)
        history_len: int = args.history_len
        hidden_size: int = args.hidden_size
        batch_size: int = args.batch_size
        device: str = args.device

        geometry = self.ensure_geometry(args.geometry_folder)

        # Create an instance of the dataset
        train_dataset = AddBiomechanicsDataset(
            os.path.abspath('../data/train'), history_len, device=torch.device(device), geometry_folder=geometry)
        dev_dataset = AddBiomechanicsDataset(
            os.path.abspath('../data/dev'), history_len, device=torch.device(device), geometry_folder=geometry)

        # Create an instance of the model
        model = self.get_model(train_dataset, model_type, history_len, hidden_size, device)

        # Create a DataLoader to load the data in batches
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

        # The number of epochs is the number of times we want to iterate over the entire dataset during training
        epochs = 40
        # Learning rate
        learning_rate = 1e-3
        # learning_rate = 1e-1

        # Define the optimizer
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            # Iterate over the entire training dataset
            loss_evaluator = RegressionLossEvaluator(contact_forces_weight=1.)
            for i, batch in enumerate(train_dataloader):
                inputs: Dict[str, torch.Tensor]
                labels: Dict[str, torch.Tensor]
                inputs, labels = batch

                # Clear the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Compute the loss
                loss = loss_evaluator(outputs, labels)

                if i % 100 == 0:
                    print('  - Batch ' + str(i) + '/' + str(len(train_dataloader)))
                if i % 5000 == 0:
                    loss_evaluator.print_report()
                    model_path = f"{checkpoint_path}/epoch_{epoch}_batch_{i}.pt"
                    if not os.path.exists(os.path.dirname(model_path)):
                        os.makedirs(os.path.dirname(model_path))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, model_path)

                # Backward pass
                loss.backward()

                # Update the model's parameters
                optimizer.step()
            # Report training loss on this epoch
            print('Epoch ' + str(epoch) + ': ')
            print('Training Set Evaluation: ')
            loss_evaluator.print_report()

            # At the end of each epoch, evaluate the model on the dev set
            dev_loss_evaluator = RegressionLossEvaluator(contact_forces_weight=1.0)
            with torch.no_grad():
                for i, batch in enumerate(dev_dataloader):
                    if i % 100 == 0:
                        print('  - Dev Batch ' + str(i) + '/' + str(len(dev_dataloader)))
                    inputs: Dict[str, torch.Tensor]
                    labels: Dict[str, torch.Tensor]
                    inputs, labels = batch
                    outputs = model(inputs)
                    loss = dev_loss_evaluator(outputs, labels)
            # Report dev loss on this epoch
            print('Dev Set Evaluation: ')
            dev_loss_evaluator.print_report()
        return True