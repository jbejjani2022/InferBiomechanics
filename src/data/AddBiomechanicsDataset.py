import nimblephysics as nimble
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os
import numpy as np


class InputDataKeys:
    # These are the joint quantities for the joints that we are observing
    POS = 'pos'
    VEL = 'vel'
    ACC = 'acc'

    # The location of the joint centers, in the root frame
    JOINT_CENTERS_IN_ROOT_FRAME = 'jointCentersInRootFrame'

    # Root velocity and acceleration, in the root frame
    ROOT_LINEAR_VEL_IN_ROOT_FRAME = 'rootLinearVelInRootFrame'
    ROOT_ANGULAR_VEL_IN_ROOT_FRAME = 'rootAngularVelInRootFrame'
    ROOT_LINEAR_ACC_IN_ROOT_FRAME = 'rootLinearAccInRootFrame'
    ROOT_ANGULAR_ACC_IN_ROOT_FRAME = 'rootAngularAccInRootFrame'

    # Recent history of the root position and orientation, in the root frame
    ROOT_POS_HISTORY_IN_ROOT_FRAME = 'rootPosHistoryInRootFrame'
    ROOT_EULER_HISTORY_IN_ROOT_FRAME = 'rootEulerHistoryInRootFrame'


class OutputDataKeys:
    TAU = 'tau'

    # These are enough to compute ID
    GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME = 'groundContactWrenchesInRootFrame'
    RESIDUAL_WRENCH_IN_ROOT_FRAME = 'residualWrenchInRootFrame'

    # These are various other things we might want to predict
    CONTACT = 'contact'
    COM_ACC_IN_ROOT_FRAME = 'comAccInRootFrame'
    GROUND_CONTACT_COPS_IN_ROOT_FRAME = 'groundContactCenterOfPressureInRootFrame'
    GROUND_CONTACT_TORQUES_IN_ROOT_FRAME = 'groundContactTorqueInRootFrame'
    GROUND_CONTACT_FORCES_IN_ROOT_FRAME = 'groundContactForceInRootFrame'


class AddBiomechanicsDataset(Dataset):
    data_path: str
    window_size: int
    geometry_folder: str
    device: torch.device
    subjects: List[nimble.biomechanics.SubjectOnDisk]
    windows: List[Tuple[int, nimble.biomechanics.SubjectOnDisk, int, int, str]]
    num_dofs: int
    num_joints: int
    contact_bodies: List[str]
    # For each subject, we store the skeleton and the contact bodies in memory, so they're ready to use with Nimble
    skeletons: List[nimble.dynamics.Skeleton]
    skeletons_contact_bodies: List[List[nimble.dynamics.BodyNode]]

    def __init__(self,
                 data_path: str,
                 window_size: int,
                 geometry_folder: str,
                 device: torch.device = torch.device('cpu'),
                 testing_with_short_dataset: bool = False,
                 skip_loading_skeletons: bool = False):
        self.data_path = data_path
        self.window_size = window_size
        self.geometry_folder = geometry_folder
        self.device = device
        self.subjects = []
        self.windows = []
        self.contact_bodies = []
        self.skeletons = []
        self.skeletons_contact_bodies = []

        # Walk the folder path, and check for any with the ".bin" extension (indicating that they are AddBiomechanics binary data files)
        num_skipped = 0
        subject_paths = []
        if os.path.isdir(data_path):
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith(".b3d"):
                        subject_paths.append(os.path.join(root, file))
        else:
            assert data_path.endswith(".b3d")
            subject_paths.append(data_path)

        if testing_with_short_dataset:
            subject_paths = subject_paths[:2]

        for i, subject_path in enumerate(subject_paths):
            # Create a subject object for each file. This will load just the header from this file, and keep that around in memory
            subject = nimble.biomechanics.SubjectOnDisk(
                subject_path)
            subject_index = len(self.subjects)
            # Add the subject to the list of subjects
            self.subjects.append(subject)
            # Get the number of degrees of freedom for this subject
            self.num_dofs = subject.getNumDofs()
            # Get the number of joints for this subject
            self.num_joints = subject.getNumJoints()
            # Get the contact bodies for this subject, and put them into a consistent order for the dataset
            contact_bodies = subject.getGroundForceBodies()
            for body in contact_bodies:
                if body == 'pelvis':
                    continue
                if body not in self.contact_bodies:
                    self.contact_bodies.append(body)
            # Also, count how many random windows we could select from this subject
            for trial in range(subject.getNumTrials()):
                probably_missing: List[bool] = [reason != nimble.biomechanics.MissingGRFReason.notMissingGRF for reason in subject.getMissingGRF(trial)]

                trial_length = subject.getTrialLength(trial)
                # print(trial_length, window_size, stride)
                # print(max(trial_length - (window_size * stride) + 1, 0))
                for window_start in range(max(trial_length - window_size + 1, 0)):
                    # Check if any of the frames in this window are probably missing GRF data
                    # If so, skip this window
                    skip = False
                    for i in range(window_start, window_start + window_size):
                        if probably_missing[i]:
                            skip = True
                            break
                    if not skip:
                        self.windows.append(
                            (subject_index, subject, trial, window_start, subject_path))
                    else:
                        num_skipped += 1

        if not skip_loading_skeletons:
            for i, subject in enumerate(self.subjects):
                # Add the skeleton to the list of skeletons
                skeleton = subject.readSkel(subject.getNumProcessingPasses()-1, geometry_folder)
                print('Loading skeleton ' + str(i+1) + '/' + str(len(subject_paths)))
                self.skeletons.append(skeleton)
                self.skeletons_contact_bodies.append([skeleton.getBodyNode(body) for body in self.contact_bodies])

        print('Contact bodies: '+str(self.contact_bodies))

        if testing_with_short_dataset:
            self.windows = self.windows[:100]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]:
        subject_index, subject, trial, start_frame, subject_path = self.windows[index]
        frames: List[nimble.biomechanics.Frame] = subject.readFrames(
            trial, start_frame, numFramesToRead=self.window_size, contactThreshold=0.1)
        dt = subject.getTrialTimestep(trial)

        # Convert the frames to a dictionary of matrices, where columns are timesteps and rows are degrees of freedom / dimensions
        # (the DataLoader will then convert this to a batched tensor)

        # Set the random seed to the index, so noise is exactly reproducible every time we retrieve this frame of data
        np.random.seed(index)

        # We first assemble the data into numpy arrays, and then convert to tensors, to save from spurious memory copies which slow down data loading
        numpy_input_dict: Dict[str, np.ndarray] = {}
        numpy_output_dict: Dict[str, np.ndarray] = {}

        # If we want to use the DYNAMICS pass, we should take the final pass as input. If we want to predict from
        # KINEMATICS instead, we should take the first pass as input
        input_pass_index = 0

        numpy_input_dict[InputDataKeys.POS] = np.row_stack([frame.processingPasses[input_pass_index].pos for frame in frames])
        numpy_input_dict[InputDataKeys.VEL] = np.row_stack([frame.processingPasses[input_pass_index].vel for frame in frames])
        numpy_input_dict[InputDataKeys.ACC] = np.row_stack([frame.processingPasses[input_pass_index].acc for frame in frames])
        numpy_input_dict[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME] = np.row_stack([frame.processingPasses[input_pass_index].jointCentersInRootFrame for frame in frames])
        numpy_input_dict[InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME] = np.row_stack([frame.processingPasses[input_pass_index].rootLinearVelInRootFrame for frame in frames])
        numpy_input_dict[InputDataKeys.ROOT_ANGULAR_VEL_IN_ROOT_FRAME] = np.row_stack([frame.processingPasses[input_pass_index].rootAngularVelInRootFrame for frame in frames])
        numpy_input_dict[InputDataKeys.ROOT_LINEAR_ACC_IN_ROOT_FRAME] = np.row_stack([frame.processingPasses[input_pass_index].rootLinearAccInRootFrame for frame in frames])
        numpy_input_dict[InputDataKeys.ROOT_ANGULAR_ACC_IN_ROOT_FRAME] = np.row_stack([frame.processingPasses[input_pass_index].rootAngularAccInRootFrame for frame in frames])
        numpy_input_dict[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME] = np.row_stack([frame.processingPasses[input_pass_index].rootPosHistoryInRootFrame for frame in frames])
        numpy_input_dict[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME] = np.row_stack([frame.processingPasses[input_pass_index].rootEulerHistoryInRootFrame for frame in frames])

        mass = subject.getMassKg()
        contact_indices: List[int] = [subject.getGroundForceBodies().index(body) if body in subject.getGroundForceBodies() else -1 for body in self.contact_bodies]
        contact_wrenches: List[np.ndarray] = []
        contact_cops: List[np.ndarray] = []
        contact_moments: List[np.ndarray] = []
        contact_forces: List[np.ndarray] = []
        for frame in frames:
            contact_wrench = np.zeros(6*len(self.contact_bodies))
            contact_cop = np.zeros(3*len(self.contact_bodies))
            contact_moment = np.zeros(3*len(self.contact_bodies))
            contact_force = np.zeros(3*len(self.contact_bodies))
            for i in range(len(self.contact_bodies)):
                if contact_indices[i] >= 0:
                    contact_wrench[6*i:6*i+6] = frame.processingPasses[-1].groundContactWrenchesInRootFrame[6*contact_indices[i]:6*contact_indices[i]+6]
                    contact_cop[3*i:3*i+3] = frame.processingPasses[-1].groundContactCenterOfPressureInRootFrame[3*contact_indices[i]:3*contact_indices[i]+3]
                    contact_moment[3*i:3*i+3] = frame.processingPasses[-1].groundContactTorqueInRootFrame[3*contact_indices[i]:3*contact_indices[i]+3]
                    contact_force[3*i:3*i+3] = frame.processingPasses[-1].groundContactForceInRootFrame[3*contact_indices[i]:3*contact_indices[i]+3]
            contact_wrenches.append(contact_wrench / mass)
            contact_cops.append(contact_cop)
            contact_moments.append(contact_moment / mass)
            contact_forces.append(contact_force / mass)
        numpy_output_dict[OutputDataKeys.TAU] = np.row_stack([frame.processingPasses[input_pass_index].tau for frame in frames])
        numpy_output_dict[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME] = np.row_stack(contact_wrenches)
        numpy_output_dict[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME] = np.row_stack(contact_cops)
        numpy_output_dict[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME] = np.row_stack(contact_moments)
        numpy_output_dict[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME] = np.row_stack(contact_forces)
        numpy_output_dict[OutputDataKeys.RESIDUAL_WRENCH_IN_ROOT_FRAME] = np.row_stack([np.array(frame.processingPasses[-1].residualWrenchInRootFrame / mass, dtype=np.float64) for frame in frames])
        numpy_output_dict[OutputDataKeys.COM_ACC_IN_ROOT_FRAME] = np.row_stack([np.array(frame.processingPasses[-1].comAccInRootFrame, dtype=np.float64) for frame in frames])

        # print(f"{numpy_output_dict[OutputDataKeys.CONTACT_FORCES]=}")
        # ###################################################
        # # Plotting
        # import matplotlib.pyplot as plt
        # x = np.arange(self.window_size)
        # # plotting each row
        # for i in range(len(self.input_dofs)):
        #     # plt.plot(x, numpy_input_dict[InputDataKeys.POS][i, :], label='pos_'+self.input_dofs[i])
        #     plt.plot(x, numpy_input_dict[InputDataKeys.VEL][i, :], label='vel_' + self.input_dofs[i])
        #     plt.plot(x, numpy_input_dict[InputDataKeys.ACC][i, :], label='acc_' + self.input_dofs[i])
        # for i in range(3):
        #     plt.plot(x, numpy_input_dict[InputDataKeys.COM_ACC][i, :], label='com_acc_' + str(i))
        # # Add the legend outside the plot
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.show()
        # ###################################################

        # Doing things inside torch.no_grad() suppresses warnings and gradient tracking
        with torch.no_grad():
            input_dict: Dict[str, torch.Tensor] = {}
            for key in numpy_input_dict:
                input_dict[key] = torch.tensor(
                    numpy_input_dict[key], dtype=torch.float32, device=self.device)

            label_dict: Dict[str, torch.Tensor] = {}
            for key in numpy_output_dict:
                label_dict[key] = torch.tensor(
                    numpy_output_dict[key], dtype=torch.float32, device=self.device)

        # Return the input and output dictionaries at this timestep, as well as the skeleton pointer

        return input_dict, label_dict, subject_index
