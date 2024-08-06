import torch
import torch.nn as nn
from typing import Dict, List
from data.AddBiomechanicsDataset import InputDataKeys, OutputDataKeys
import logging


ACTIVATION_FUNCS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid()
}


class FeedForwardBaseline(nn.Module):
    num_dofs: int
    num_contact_bodies: int
    history_len: int
    root_history_len: int

    def __init__(self,
                 num_dofs: int,
                 num_contact_bodies: int,
                 history_len: int,
                 output_data_format: str,
                 activation: str,
                 stride: int,
                 root_history_len: int,
                 hidden_dims: List[int] = [512],
                 batchnorm: bool = False,
                 dropout: bool = False,
                 dropout_prob: float = 0.0,
                 device: str = 'cpu'):
        super(FeedForwardBaseline, self).__init__()
        self.stride = stride
        self.activation = activation
        self.output_data_format = output_data_format
        self.num_dofs = num_dofs
        self.num_contact_bodies = num_contact_bodies
        self.history_len = history_len
        self.root_history_len = root_history_len
        self.device = device
        
        # print('num dofs: ' + str(num_dofs) + ', num contact bodies: ' + str(num_contact_bodies)+ ', history len: ' + str(history_len), ', root history len: ' + str(root_history_len), f"{stride=}")

        # Compute input and output sizes
        # 10 input features
        # For each frame in the window:
        # We need each DoF for each of position, velocity, and acceleration
        # Need x, y, z coors for each of rootLinearVelInRootFrame, rootLinearAccInRootFrame, rootAngularVelInRootFrame, and rootAngularAccInRootFrame
        # For rootPosHistoryInRootFrame and rootEulerHistoryInRootFrame: each is the concatention of `stride` number of 3 vectors — representing the 'recent' history
        # For jointCentersInRootFrame: need x, y, z coors for each of 12 joints
        self.input_size = (3 * num_dofs + 4 * 3 + 2 * stride * 3 + 12 * 3) * (history_len // stride)  # = 1470 (for 23 dofs and window size 10)
        print(f'input size = {self.input_size}')

        # 4 output features
        # For each output frame, need:
        # groundContactCenterOfPressureInRootFrame - concatenated 3 vectors for each contact body. Each 3 vector represents center of pressure (CoP) for a contact measured on the force plate.
        # groundContactForceInRootFrame - same as above, but ground-reaction force instead of CoP
        # groundContactTorqueInRootFrame - same, but torque instead of CoP
        # groundContactWrenchesInRootFrame - a wrench is a vector of length 6, composed of first 3 = torque, last 3 = force. One wrench per contact body.
        self.num_output_frames = (history_len // stride) if output_data_format == 'all_frames' else 1
        self.output_size = num_contact_bodies * (3 * 3 + 6) * self.num_output_frames
        print(f'output size = {self.output_size}')  # = 300 (for 2 contact bodies and 10 output frames)

        self.net = []
        dims = [self.input_size] + hidden_dims + [self.output_size]
        print(f"MODEL DIMENSIONS: input size = {self.input_size}, hidden dims = {hidden_dims}, output size = {self.output_size}")
        for i, (h0, h1) in enumerate(zip(dims[:-1], dims[1:])):
            if dropout:
                self.net.append(nn.Dropout(dropout_prob))
            if batchnorm:
                self.net.append(nn.BatchNorm1d(h0))
            self.net.append(nn.Linear(h0, h1, dtype=torch.float32, device=self.device))
            if i < len(dims) - 2:
                self.net.append(ACTIVATION_FUNCS[self.activation])
        
        self.net = nn.Sequential(*self.net)
        logging.info(f"{self.net=}")
        
    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. Check input shape matches our assumptions.
        # shape is (B, T, C) - batches, timesteps, channels
        assert len(input[InputDataKeys.POS].shape) == 3
        assert input[InputDataKeys.POS].shape[-1] == self.num_dofs
        assert len(input[InputDataKeys.VEL].shape) == 3
        assert input[InputDataKeys.VEL].shape[-1] == self.num_dofs
        assert len(input[InputDataKeys.ACC].shape) == 3
        assert input[InputDataKeys.ACC].shape[-1] == self.num_dofs
        assert len(input[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME].shape) == 3
        assert input[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME].shape[-1] == 12 * 3
        assert len(input[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME].shape) == 3
        assert input[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME].shape[-1] == self.stride * 3
        assert len(input[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME].shape) == 3
        assert input[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME].shape[-1] == self.stride * 3

        # 2. Concatenate the inputs together and flatten them into a single vector for all timesteps
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
        ], dim=-1).reshape((input[InputDataKeys.POS].shape[0], -1)).to(self.device)

        batch_size = inputs.shape[0]
        
        # 3. Actually run the forward pass
        x = self.net(inputs)

        # 4. Break up the output vector into the different requested components
        return {
            OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME: x[:, 0*self.num_output_frames:6*self.num_output_frames].reshape((batch_size, self.num_output_frames, 6)),
            OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME: x[:, 6*self.num_output_frames:12*self.num_output_frames].reshape((batch_size, self.num_output_frames, 6)),
            OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME: x[:, 12*self.num_output_frames:18*self.num_output_frames].reshape((batch_size, self.num_output_frames, 6)),
            OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME: x[:, 18*self.num_output_frames:30*self.num_output_frames].reshape((batch_size, self.num_output_frames, 12))
        }