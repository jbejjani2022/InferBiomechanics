import argparse
import os
import torch
from models.FeedForwardRegressionBaseline import FeedForwardBaseline
from models.Groundlink import Groundlink
from models.AnalyticalBaseline import AnalyticalBaseline
from models.MDM import MDM
from data.AddBiomechanicsDataset import AddBiomechanicsDataset
from typing import List
import logging


class AbstractCommand:
    """
    All of our different activities inherit from this class. This class defines the interface for a CLI command, so
    that it's convenient to split commands across files. It also carries shared logic for loading / saving models, etc.
    """
    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        pass

    def run(self, args: argparse.Namespace) -> bool:
        pass

    def register_model_options(self, parser: argparse.ArgumentParser):
        pass

    def ensure_geometry(self, geometry: str):
        if geometry is None:
            # Check if the "./Geometry" folder exists, and if not, download it
            if not os.path.exists('./Geometry'):
                print('Downloading the Geometry folder from https://addbiomechanics.org/resources/Geometry.zip')
                exit_code = os.system('wget https://addbiomechanics.org/resources/Geometry.zip')
                if exit_code != 0:
                    print('ERROR: Failed to download Geometry.zip. You may need to install wget. If you are on a Mac, '
                          'try running "brew install wget"')
                    return False
                os.system('unzip ./Geometry.zip')
                os.system('rm ./Geometry.zip')
            geometry = './Geometry'
        print('Using Geometry folder: ' + geometry)
        geometry = os.path.abspath(geometry)
        if not geometry.endswith('/'):
            geometry += '/'
        return geometry
    
    def get_model(self,
                  num_dofs: int,
                  num_joints: int,
                  model_type: str = 'feedforward',
                  history_len: int = 5,
                  stride: int = 1,
                  hidden_dims: List[int] = [512],
                  activation: str = 'relu',
                  batchnorm: bool = False,
                  dropout: bool = False,
                  dropout_prob: float = 0.0,
                  root_history_len: int = 10,
                  output_data_format: str = 'all_frames',
                  device: str = 'cpu'):
        if model_type == 'feedforward':
            model = FeedForwardBaseline(
                num_dofs,
                num_joints,
                history_len,
                output_data_format,
                activation,
                stride=stride,
                hidden_dims=hidden_dims,
                batchnorm=batchnorm,
                dropout=dropout,
                dropout_prob=dropout_prob,
                root_history_len=root_history_len,
                device=device
            )
        elif model_type == 'groundlink':
            model = Groundlink(
                num_dofs,
                num_joints,
                root_history_len,
                output_data_format
            )
        elif model_type == 'mdm':
            model = MDM(
                num_dofs
            )
        else:
            assert(model_type == 'analytical')
            model = AnalyticalBaseline()

        return model
    

    def load_latest_checkpoint(self, model, optimizer=None, checkpoint_dir="../checkpoints", map_location=None):
        if not os.path.exists(checkpoint_dir):
            print("Checkpoint directory does not exist!")
            return

        # Get all the checkpoint files
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]

        # If there are no checkpoints, return
        if not checkpoints:
            print("No checkpoints available!")
            return

        # Sort the files based on the epoch and batch numbers in their filenames
        checkpoints.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3].split('.')[0])))

        # Get the path of the latest checkpoint
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])

        logging.info(f"{latest_checkpoint=}")
        # Load the checkpoint
        checkpoint = torch.load(latest_checkpoint, map_location=map_location)

        # Load the model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # You might also want to return the epoch and batch number so you can continue training from there
        epoch = checkpoint['epoch']
        batch = checkpoints[-1].split('_')[3].split('.')[0]

        print(f"Loaded checkpoint from epoch {epoch}, batch {batch}")

        return epoch, int(batch)
