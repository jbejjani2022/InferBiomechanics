import nimblephysics as nimble
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os
import numpy as np
from typing import Optional
import argparse

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
    windows: List[Tuple[int, nimble.biomechanics.SubjectOnDisk, int, int, str]]
    num_dofs: int
    num_joints: int
    contact_bodies: List[str]
    # For each subject, we store the skeleton and the contact bodies in memory, so they're ready to use with Nimble
    skeletons: List[nimble.dynamics.Skeleton]
    skeletons_contact_bodies: List[List[nimble.dynamics.BodyNode]]

    def __init__(self,
                 args: argparse.Namespace,
                 data_path: str,
                 window_size: int,
                 geometry_folder: str,
                 device: torch.device = torch.device('cpu'),
                 testing_with_short_dataset: bool = False,
                 skip_loading_skeletons: bool = False):
        self.args = args
        self.subject_paths = []
        self.subjects = []
        self.trials = []
        self.window_size = window_size
        self.geometry_folder = geometry_folder
        self.device = device
        self.windows = []
        self.contact_bodies = []
        self.skeletons = []
        self.skeletons_contact_bodies = []
        

        if os.path.isdir(data_path):
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith(".b3d") and "vander" not in file.lower():
                        self.subject_paths.append(os.path.join(root, file))
        else:
            assert data_path.endswith(".b3d")
            self.subject_paths.append(data_path)
        
        if testing_with_short_dataset:
            self.subject_paths = self.subject_paths[11:12]
        self.subject_indices = {subject_path: i for i, subject_path in enumerate(self.subject_paths)}

        # Walk the folder path, and check for any with the ".bin" extension (indicating that they are AddBiomechanics binary data files)
        if len(self.subject_paths) > 0:
            # Create a subject object for each file. This will load just the header from this file, and keep that around in memory
            subject = nimble.biomechanics.SubjectOnDisk(
                self.subject_paths[0])
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


        if not skip_loading_skeletons:
            for i, subject_path in enumerate(self.subject_paths):
                # Add the skeleton to the list of skeletons
                subject = nimble.biomechanics.SubjectOnDisk(subject_path)
                skeleton = subject.readSkel(subject.getNumProcessingPasses()-1, geometry_folder)
                print('Loading skeleton ' + str(i+1) + '/' + str(len(self.subject_paths)) + f'for subject {subject_path}')
                self.subjects.append(subject)
                self.skeletons.append(skeleton)
                print(f"{[subject.getTrialName(trial_id) for trial_id in range(subject.getNumTrials())]}")
                self.trials.extend([(subject_path, trial_id) for trial_id in range(subject.getNumTrials()) if any(x in subject.getTrialName(trial_id).lower() for x in self.args.trial_filter)])
                self.skeletons_contact_bodies.append([skeleton.getBodyNode(body) for body in self.contact_bodies])
	
        print(f"{self.trials=}, {len(self.trials)=}")
        print('Contact bodies: '+str(self.contact_bodies))

    def prepare_data_for_subset(self, trial_indices: Optional[List[int]] = None):
        self.windows = []
        num_skipped = 0

        if trial_indices is None:
            trial_indices = range(len(self.trials))

        print('Pre-loading data for subset of trials '+str(trial_indices))

        for progress_index, trial_index in enumerate(trial_indices):
            print('Pre-loading subject '+str(progress_index+1)+'/'+str(len(trial_indices)))
            np.random.seed(trial_index)
            subject_path, trial = self.trials[trial_index]
            subject_index = self.subject_indices[subject_path]
            subject = nimble.biomechanics.SubjectOnDisk(subject_path)

            # Pre-load all the frames of this trial
            subject.loadAllFrames(doNotStandardizeForcePlateData=True)

            probably_missing: List[bool] = [reason != nimble.biomechanics.MissingGRFReason.notMissingGRF for reason in subject.getMissingGRF(trial)]

            trial_length = subject.getTrialLength(trial)

            processing_passes: List[nimble.biomechanics.SubjectOnDiskTrialPass] = subject.getHeaderProto().getTrials()[trial].getPasses()

            # If we want to use the DYNAMICS pass, we should take the final pass as input. If we want to predict from
            # KINEMATICS instead, we should take the first pass as input
            input_pass_index = 0

            # Copy all of the Numpy arrays from C++ to Python only once, because that's a slow operation

            # Inputs
            pos = processing_passes[input_pass_index].getPoses().transpose().astype(np.float32)
            vel = processing_passes[input_pass_index].getVels().transpose().astype(np.float32)
            acc = processing_passes[input_pass_index].getAccs().transpose().astype(np.float32)
            joint_centers_in_root_frame = processing_passes[input_pass_index].getJointCentersInRootFrame().transpose().astype(np.float32)
            root_spatial_vel_in_root_frame = processing_passes[input_pass_index].getRootSpatialVelInRootFrame().transpose().astype(np.float32)
            root_spatial_acc_in_root_frame = processing_passes[input_pass_index].getRootSpatialAccInRootFrame().transpose().astype(np.float32)
            root_pos_history = processing_passes[input_pass_index].getRootPosHistoryInRootFrame().transpose().astype(np.float32)
            root_euler_history = processing_passes[input_pass_index].getRootEulerHistoryInRootFrame().transpose().astype(np.float32)

            # Outputs
            tau = processing_passes[-1].getTaus().transpose().astype(np.float32)
            ground_contact_wrenches_in_root_frame = processing_passes[-1].getGroundBodyWrenchesInRootFrame().transpose().astype(np.float32)
            ground_contact_cop_torque_force_in_root_frame = processing_passes[-1].getGroundBodyCopTorqueForceInRootFrame().transpose().astype(np.float32)
            com_acc_in_root_frame = processing_passes[-1].getComAccsInRootFrame().transpose().astype(np.float32)
            residual_wrench_in_root_frame = processing_passes[-1].getResidualWrenchInRootFrame().transpose().astype(np.float32)

            for window_start in range(max(trial_length - self.window_size + 1, 0)):
                # Check if any of the frames in this window are probably missing GRF data
                # If so, skip this window
                skip = False
                for i in range(window_start, window_start + self.window_size):
                    if probably_missing[i]:
                        skip = True
                        break
                if not skip:
                    window_end_exclusive = window_start + self.window_size
                    # We first assemble the data into numpy arrays, and then convert to tensors, to save from spurious memory copies which slow down data loading
                    numpy_input_dict: Dict[str, np.ndarray] = {}
                    numpy_output_dict: Dict[str, np.ndarray] = {}

                    numpy_input_dict[InputDataKeys.POS] = pos[window_start:window_end_exclusive, :]
                    numpy_input_dict[InputDataKeys.VEL] = vel[window_start:window_end_exclusive, :]
                    numpy_input_dict[InputDataKeys.ACC] = acc[window_start:window_end_exclusive, :]
                    numpy_input_dict[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME] = joint_centers_in_root_frame[window_start:window_end_exclusive, :]
                    numpy_input_dict[InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME] = root_spatial_vel_in_root_frame[window_start:window_end_exclusive, 3:6]
                    numpy_input_dict[InputDataKeys.ROOT_ANGULAR_VEL_IN_ROOT_FRAME] = root_spatial_vel_in_root_frame[window_start:window_end_exclusive, 0:3]
                    numpy_input_dict[InputDataKeys.ROOT_LINEAR_ACC_IN_ROOT_FRAME] = root_spatial_acc_in_root_frame[window_start:window_end_exclusive, 3:6]
                    numpy_input_dict[InputDataKeys.ROOT_ANGULAR_ACC_IN_ROOT_FRAME] = root_spatial_acc_in_root_frame[window_start:window_end_exclusive, 0:3]
                    numpy_input_dict[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME] = root_pos_history[window_start:window_end_exclusive, :]
                    numpy_input_dict[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME] = root_euler_history[window_start:window_end_exclusive, :]

                    mass = subject.getMassKg()
                    numpy_output_dict[OutputDataKeys.TAU] = tau[window_start:window_end_exclusive, :]
                    numpy_output_dict[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME] = np.zeros((6*len(self.contact_bodies)), dtype=np.float32)
                    numpy_output_dict[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME] = np.zeros((3*len(self.contact_bodies)), dtype=np.float32)
                    numpy_output_dict[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME] = np.zeros((3*len(self.contact_bodies)), dtype=np.float32)
                    numpy_output_dict[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME] = np.zeros((3*len(self.contact_bodies)), dtype=np.float32)
                    contact_indices: List[int] = [subject.getGroundForceBodies().index(body) if body in subject.getGroundForceBodies() else -1 for body in self.contact_bodies]
                    for i in range(len(self.contact_bodies)):
                        if contact_indices[i] >= 0:
                            numpy_output_dict[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME][6*i:6*i+6] = ground_contact_wrenches_in_root_frame[window_end_exclusive-1, 6*contact_indices[i]:6*contact_indices[i]+6] / mass
                            numpy_output_dict[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME][3*i:3*i+3] = ground_contact_cop_torque_force_in_root_frame[window_end_exclusive-1, 9*contact_indices[i]:9*contact_indices[i]+3]
                            numpy_output_dict[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME][3*i:3*i+3] = ground_contact_cop_torque_force_in_root_frame[window_end_exclusive-1, 9*contact_indices[i]+3:9*contact_indices[i]+6] / mass
                            numpy_output_dict[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME][3*i:3*i+3] = ground_contact_cop_torque_force_in_root_frame[window_end_exclusive-1, 9*contact_indices[i]+6:9*contact_indices[i]+9] / mass
                    numpy_output_dict[OutputDataKeys.RESIDUAL_WRENCH_IN_ROOT_FRAME] = residual_wrench_in_root_frame[window_end_exclusive-1, :]
                    numpy_output_dict[OutputDataKeys.COM_ACC_IN_ROOT_FRAME] = com_acc_in_root_frame[window_end_exclusive-1, :]

                    # Doing things inside torch.no_grad() suppresses warnings and gradient tracking
                    with torch.no_grad():
                        input_dict: Dict[str, torch.Tensor] = {}
                        for key in numpy_input_dict:
                            input_dict[key] = torch.from_numpy(
                                numpy_input_dict[key]).detach()

                        label_dict: Dict[str, torch.Tensor] = {}
                        for key in numpy_output_dict:
                            label_dict[key] = torch.from_numpy(
                                numpy_output_dict[key]).detach()
                    
                    self.windows.append((input_dict, label_dict, subject_index, trial))
                else:
                    num_skipped += 1

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]:
        input_dict, label_dict, subject_index, trial = self.windows[index]
        # Convert the frames to a dictionary of matrices, where columns are timesteps and rows are degrees of freedom / dimensions
        # (the DataLoader will then convert this to a batched tensor)

        # Set the random seed to the index, so noise is exactly reproducible every time we retrieve this frame of data

        

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

        

        # Return the input and output dictionaries at this timestep, as well as the skeleton pointer

        return input_dict, label_dict, subject_index, trial
