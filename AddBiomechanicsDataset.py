import nimblephysics as nimble
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os
import numpy as np


class InputDataKeys:
    POS = 'pos'
    VEL = 'vel'
    ACC = 'acc'
    COM_ACC = 'com_acc'


class OutputDataKeys:
    CONTACT = 'contact'
    COM_ACC = 'com_acc'
    CONTACT_FORCES = 'contact_forces'


class AddBiomechanicsDataset(Dataset):
    data_path: str
    window_size: int
    stride: int
    device: torch.device
    subjects: List[nimble.biomechanics.SubjectOnDisk]
    input_dofs: List[str]
    input_dof_indices: List[int]
    windows: List[Tuple[nimble.biomechanics.SubjectOnDisk, int, int, str]]

    def __init__(self, data_path: str, window_size: int, stride: int, input_dofs: List[str], device: torch.device = torch.device('cpu')):
        self.data_path = data_path
        self.window_size = window_size
        self.stride = stride
        self.input_dofs = input_dofs
        self.device = device
        self.subjects = []
        self.windows = []

        # Walk the folder path, and check for any with the ".bin" extension (indicating that they are AddBiomechanics binary data files)
        num_skipped = 0
        subject_paths = []
        if os.path.isdir(data_path):
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith(".bin"):
                        subject_paths.append(os.path.join(root, file))
        else:
            assert data_path.endswith(".bin")
            subject_paths.append(data_path)

        for subject_path in subject_paths:
            # Create a subject object for each file. This will load just the header from this file, and keep that around in memory
            subject = nimble.biomechanics.SubjectOnDisk(
                subject_path)
            # Add the subject to the list of subjects
            self.subjects.append(subject)
            # Also, count how many random windows we could select from this subject
            for trial in range(subject.getNumTrials()):
                probably_missing: List[bool] = subject.getProbablyMissingGRF(trial)

                trial_length = subject.getTrialLength(trial)
                # print(trial_length, window_size, stride)
                # print(max(trial_length - (window_size * stride) + 1, 0))
                for window_start in range(max(trial_length - (window_size * stride) + 1, 0)):
                    # Check if any of the frames in this window are probably missing GRF data
                    # If so, skip this window
                    skip = False
                    for i in range(window_start, window_start + window_size):
                        if probably_missing[i]:
                            skip = True
                            break
                    if not skip:
                        self.windows.append(
                            (subject, trial, window_start, subject_path))
                    else:
                        num_skipped += 1

        # Read the dofs from the first subject (assuming they are all the same)
        self.input_dof_indices = []
        skel = self.subjects[0].readSkel()
        dof_names = []
        for i in range(skel.getNumDofs()):
            dof_name = skel.getDofByIndex(i).getName()
            dof_names.append(dof_name)

        for dof_name in input_dofs:
            index = dof_names.index(dof_name)
            if index >= 0:
                self.input_dof_indices.append(index)
            else:
                # Throw an exception
                raise Exception('Dof ' + dof_name + ' not found in input dofs')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int):
        subject, trial, start_frame, subject_path = self.windows[index]
        frames: List[nimble.biomechanics.Frame] = subject.readFrames(
            trial, start_frame, numFramesToRead=self.window_size, stride=self.stride, contactThreshold=0.1)
        dt = subject.getTrialTimestep(trial)

        # Convert the frames to a dictionary of matrices, where columns are timesteps and rows are degrees of freedom / dimensions
        # (the DataLoader will then convert this to a batched tensor)

        # Set the random seed to the index, so noise is exactly reproducible every time we retrieve this frame of data
        np.random.seed(index)

        # We first assemble the data into numpy arrays, and then convert to tensors, to save from spurious memory copies which slow down data loading
        numpy_input_dict: Dict[str, np.ndarray] = {}
        numpy_output_dict: Dict[str, np.ndarray] = {}


        poses = [frame.pos for frame in frames]
        numpy_input_dict[InputDataKeys.POS] = np.column_stack([frame.pos[self.input_dof_indices] for frame in frames])
        numpy_input_dict[InputDataKeys.VEL] = np.column_stack([frame.vel[self.input_dof_indices] for frame in frames])
        numpy_input_dict[InputDataKeys.ACC] = np.column_stack([frame.acc[self.input_dof_indices] for frame in frames])
        numpy_input_dict[InputDataKeys.COM_ACC] = np.column_stack([frame.comAcc for frame in frames])

        numpy_output_dict[OutputDataKeys.CONTACT] = np.column_stack([np.array(frame.contact, dtype=np.float64) for frame in frames])
        correct = subject.getContactBodies()[0][-1] == 'l'
        left = 0 if correct else 1
        right = 1 - left
        contact_class = 0
        if frames[-1].contact[left] == 0 and frames[-1].contact[right] == 0:
            # Flight phase
            contact_class = 0
        elif frames[-1].contact[left] == 1 and frames[-1].contact[right] == 0:
            # Left foot stance
            contact_class = 1
        elif frames[-1].contact[left] == 0 and frames[-1].contact[right] == 1:
            # Right foot stance
            contact_class = 2
        elif frames[-1].contact[left] == 1 and frames[-1].contact[right] == 1:
            # Double stance
            contact_class = 3
        one_hot_contact = np.zeros(4, dtype=np.float32)
        one_hot_contact[contact_class] = 1

        numpy_output_dict[OutputDataKeys.CONTACT] = one_hot_contact
        numpy_output_dict[OutputDataKeys.CONTACT_FORCES] = frames[-1].groundContactForce if correct else frames[-1].groundContactForce[[3,4,5,0,1,2]]
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

        if (numpy_input_dict[InputDataKeys.POS] == 0).all():
            print("Warning: all zeros input POS")
            print('Subject path: '+subject_path)
            print('Trial: '+str(trial))
            print('Trial length: '+str(subject.getTrialLength(trial)))
            print('Start frame: '+str(start_frame))
            print('Window size: '+str(self.window_size))
            for i in range(self.window_size):
                frame = frames[i]
                print('Frame '+str(i)+': ')
                print(frame.pos)
                print('Original copy '+str(i)+': ')
                print(poses[i])

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

        # Return the input and output dictionaries at this timestep

        return input_dict, label_dict

if __name__ == "__main__":
    window_size = 50
    stride = 20
    batch_size = 32
    device = 'cpu'

    # Input dofs to train on
    input_dofs = ['knee_angle_l', 'knee_angle_r', 'hip_flexion_l', 'hip_flexion_r', 'hip_adduction_l', 'hip_adduction_r']
    data_path = "/Users/rishi/Documents/Academics/stanford/human-body-dynamics/InferBiomechanics/data/processed/standardized/rajagopal_no_arms/data/protected/us-west-2:be72ee5a-acdb-4e07-b288-a55886ca1e3b/data/c1ab/5dd9f9149f8e8064442a852d79e77050a17c772bdc1199cfe088177b9387a657/5dd9f9149f8e8064442a852d79e77050a17c772bdc1199cfe088177b9387a657.bin"
    AddBiomechanicsDataset(data_path, window_size, stride, input_dofs=input_dofs, device=torch.device(device))