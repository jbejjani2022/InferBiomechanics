import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from data.AddBiomechanicsDataset import InputDataKeys, OutputDataKeys
import nimblephysics as nimble


class FeedForwardBaseline(nn.Module):
    num_dofs: int
    num_joints: int
    history_len: int
    root_history_len: int
    hidden_size: int

    def __init__(self,
                 num_dofs: int,
                 num_joints: int,
                 history_len: int,
                 hidden_size: int = 64,
                 dropout_prob: float = 0.1,
                 device: str = 'cpu',
                 root_history_len: int = 10):
        super(FeedForwardBaseline, self).__init__()
        self.num_dofs = num_dofs
        self.num_joints = num_joints
        self.history_len = history_len
        self.root_history_len = root_history_len
        print('num dofs: ' + str(num_dofs) + ', num joints: ' + str(num_joints)+', history len: ' + str(history_len))
        self.hidden_size = hidden_size

        # Compute input and output sizes

        # For input, we need each dof, for position and velocity and acceleration, for each frame in the window, and then also the COM acceleration for each frame in the window
        input_size = (num_dofs * 3 + 12 + num_joints * 3 + root_history_len * 6)
        # For output, we have four foot-ground contact classes (foot 1, foot 2, both, neither)
        output_size = 30

        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float32, device=device)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size, dtype=torch.float32, device=device)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.sigmoid2 = nn.Sigmoid()
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size, dtype=torch.float32, device=device)
        
    def forward(self, input: Dict[str, torch.Tensor], skels_and_contact: List[Tuple[nimble.dynamics.Skeleton, List[nimble.dynamics.BodyNode]]]) -> dict[str, torch.Tensor]:
        # Get the position, velocity, and acceleration tensors

        # assert(input[InputDataKeys.POS].shape[-1] == self.num_dofs)
        # assert(input[InputDataKeys.VEL].shape[-1] == self.num_dofs)
        # assert(input[InputDataKeys.ACC].shape[-1] == self.num_dofs)
        # assert(input[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME].shape[-1] == self.num_joints * 3)
        # assert(input[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME].shape[-1] == self.root_history_len * 3)
        # assert(input[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME].shape[-1] == self.root_history_len * 3)

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
        ], dim=-1)
        # Actually run the forward pass
        x = self.dropout1(inputs)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid2(x)
        x = self.dropout3(x)
        x = self.fc3(x)

        return {
            OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME: x[:, :, 0:6],
            OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME: x[:, :, 6:12],
            OutputDataKeys.GROUND_CONTACT_MOMENTS_IN_ROOT_FRAME: x[:, :, 12:18],
            OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME: x[:, :, 18:30]
        }