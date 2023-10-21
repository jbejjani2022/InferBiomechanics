import argparse
import os
import torch
from models.FeedForwardRegressionBaseline import FeedForwardBaseline
from data.AddBiomechanicsDataset import AddBiomechanicsDataset


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
                  dataset: AddBiomechanicsDataset,
                  model_type: str = 'feedforward',
                  history_len: int = 5,
                  hidden_size: int = 512,
                  device: str = 'cpu'):
        # Define the model
        model = FeedForwardBaseline(
            dataset.num_dofs,
            dataset.num_joints,
            history_len,
            hidden_size,
            dropout_prob=0.0,
            device=device)

        return model

    def load_latest_checkpoint(self, model, optimizer=None, output_dir="./outputs/models"):
        # Get all the checkpoint files
        checkpoints = [f for f in os.listdir(output_dir) if f.endswith(".pt")]

        # If there are no checkpoints, return
        if not checkpoints:
            print("No checkpoints available!")
            return

        # Sort the files based on the epoch and batch numbers in their filenames
        checkpoints.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3].split('.')[0])))

        # Get the path of the latest checkpoint
        latest_checkpoint = os.path.join(output_dir, checkpoints[-1])

        # Load the checkpoint
        checkpoint = torch.load(latest_checkpoint)

        # Load the model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # You might also want to return the epoch and batch number so you can continue training from there
        epoch = checkpoint['epoch']
        batch = checkpoints[-1].split('_')[3].split('.')[0]

        print(f"Loaded checkpoint from epoch {epoch}, batch {batch}")

        return epoch, int(batch)
