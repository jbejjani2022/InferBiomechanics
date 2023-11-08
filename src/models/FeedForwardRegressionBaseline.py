import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from data.AddBiomechanicsDataset import InputDataKeys, OutputDataKeys
import nimblephysics as nimble
import argparse
import logging

ACTIVATION_FUNCS = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}
class FeedForwardBaseline(nn.Module):
    args: argparse.Namespace
    num_dofs: int
    num_joints: int
    history_len: int
    root_history_len: int
    per_output_nets: nn.ModuleList

    def __init__(self,
                 args: argparse.Namespace,
                 num_dofs: int,
                 num_joints: int,
                 history_len: int,
                 device: str = 'cpu',
                 root_history_len: int = 10):
        super(FeedForwardBaseline, self).__init__()
        self.args = args
        self.num_dofs = num_dofs
        self.num_joints = num_joints
        self.history_len = history_len
        self.root_history_len = root_history_len
        print('num dofs: ' + str(num_dofs) + ', num joints: ' + str(num_joints)+', history len: ' + str(history_len))

        # Compute input and output sizes

        # For input, we need each dof, for position and velocity and acceleration, for each frame in the window, and
        # then also the COM acceleration for each frame in the window
        self.input_size = (num_dofs * 3 + 12 + num_joints * 3 + root_history_len * 6) * (self.args.history_len // self.args.stride)
        # For output, we have CoPs for each foot, and forces and torques for each foot, as well as the wrenches
        # for each foot
        self.output_size = 30

        self.per_output_nets = nn.ModuleList()
        for output in range(self.output_size):
            net: List[nn.Module] = []
            dims = [self.input_size] + self.args.hidden_dims + [1]
            for i, (h0, h1) in enumerate(zip(dims[:-1], dims[1:])):
                if self.args.dropout:
                    net.append(nn.Dropout(self.args.dropout_prob))
                if self.args.batchnorm:
                    net.append(nn.BatchNorm1d(h0))
                net.append(nn.Linear(h0, h1, dtype=torch.float32, device=device))
                if i < len(dims)-2:
                    net.append(ACTIVATION_FUNCS[self.args.activation])
            self.per_output_nets.append(nn.Sequential(*net))
        logging.info(f"{self.per_output_nets=}")
        
    def forward(self, input: Dict[str, torch.Tensor], skels_and_contact: List[Tuple[nimble.dynamics.Skeleton, List[nimble.dynamics.BodyNode]]]) -> dict[str, torch.Tensor]:
        # 1. Check input shape matches our assumptions.
        assert len(input[InputDataKeys.POS].shape) == 3
        assert input[InputDataKeys.POS].shape[-1] == self.num_dofs
        assert len(input[InputDataKeys.VEL].shape) == 3
        assert input[InputDataKeys.VEL].shape[-1] == self.num_dofs
        assert len(input[InputDataKeys.ACC].shape) == 3
        assert input[InputDataKeys.ACC].shape[-1] == self.num_dofs
        assert len(input[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME].shape) == 3
        assert input[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME].shape[-1] == self.num_joints * 3
        assert len(input[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME].shape) == 3
        assert input[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME].shape[-1] == self.root_history_len * 3
        assert len(input[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME].shape) == 3
        assert input[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME].shape[-1] == self.root_history_len * 3

        # 2. Concatenate inputs, and flatten them to be a single long vector for each batch entry. That means each
        # timestep in the input data gets concatenated end-to-end.
        inputs = torch.concat([
            input[InputDataKeys.POS],
            input[InputDataKeys.VEL],
            input[InputDataKeys.ACC],
            input[InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME],
            input[InputDataKeys.ROOT_ANGULAR_VEL_IN_ROOT_FRAME],
            input[InputDataKeys.ROOT_LINEAR_ACC_IN_ROOT_FRAME],
            input[InputDataKeys.ROOT_ANGULAR_ACC_IN_ROOT_FRAME],
            input[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME],
            input[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME],
            input[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME]
        ], dim=-1).reshape((input[InputDataKeys.POS].shape[0], -1))

        # 3. Run one forward model stack per output variable. Each stack is an MLP. The reason to have separate stacks
        # per output is that we are doing a regression with different scales for each output, and we don't want to
        # have to worry about scaling the outputs to be the same scale. So, we just train separate models for each
        # output variable.
        x = torch.cat([net(inputs) for net in self.per_output_nets], dim=-1)

        # 4. Break the output back up into separate tensors for each output variable. Predicts just a single frame of
        # output, given the input context.
        return {
            OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME: x[:, 0:6],
            OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME: x[:, 6:12],
            OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME: x[:, 12:18],
            OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME: x[:, 18:30]
        }