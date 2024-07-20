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

from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps


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
    
    def get_gaussian_diffusion(args):
    # default params
        predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
        steps = args.diffusion_steps
        scale_beta = 1.  # no scaling
        timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
        learn_sigma = False
        rescale_timesteps = False

        betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
        loss_type = gd.LossType.MSE

        if not timestep_respacing:
            timestep_respacing = [steps]

        return SpacedDiffusion(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not args.sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            lambda_vel=args.lambda_vel,
            lambda_rcxyz=args.lambda_rcxyz,
            lambda_fc=args.lambda_fc,
        )

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
        checkpoint = torch.load(latest_checkpoint)

        # Load the model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'], map_location=map_location)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # You might also want to return the epoch and batch number so you can continue training from there
        epoch = checkpoint['epoch']
        batch = checkpoints[-1].split('_')[3].split('.')[0]

        print(f"Loaded checkpoint from epoch {epoch}, batch {batch}")

        return epoch, int(batch)
