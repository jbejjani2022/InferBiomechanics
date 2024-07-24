import torch
import numpy as np

from torch.utils.data import DataLoader
from typing import Dict, List

from test.get_dataset import get_dataset


short = True
foot_contact = True
history_len = 50
train_dataset, dev_dataset = get_dataset(short, foot_contact, history_len)

batch_size = 1
data_loading_workers = 1
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=data_loading_workers, persistent_workers=True)

LEFT = 0
RIGHT = 1


def process_inputs(inputs):
    # contact data from each frame of first subject in inputs dict
    contact = inputs['contact'][0]
    # exclude last frame to facilitate velocity calculation
    contact = contact[:-1]
    # indices of frames in which left foot is in contact with ground
    left_contact = [i for i in range(len(contact)) if contact[i][LEFT]]
    # indices of frames in which left foot is in contact with ground
    right_contact = [i for i in range(len(contact)) if contact[i][RIGHT]]
    
    # velocities of subtala left/right joints for frames where left/right foot in contact
    subtalar = inputs['subtalar_angles'][0]
    # subtalar_l_vel = [subtalar[i + 1][LEFT] - subtalar[i][LEFT] for i in left_contact]
    # subtalar_r_vel = [subtalar[i + 1][RIGHT] - subtalar[i][RIGHT] for i in right_contact]
    subtalar_l_vel = [subtalar[i + 1][LEFT] - subtalar[i][LEFT] for i in range(len(contact))]
    subtalar_r_vel = [subtalar[i + 1][RIGHT] - subtalar[i][RIGHT] for i in range(len(contact))]
    
    mtp = inputs['mtp_angles'][0]
    # mtp_l_vel = [mtp[i + 1][LEFT] - mtp[i][LEFT] for i in left_contact]
    # mtp_r_vel = [mtp[i + 1][RIGHT] - mtp[i][RIGHT] for i in right_contact]
    mtp_l_vel = [mtp[i + 1][LEFT] - mtp[i][LEFT] for i in range(len(contact))]
    mtp_r_vel = [mtp[i + 1][RIGHT] - mtp[i][RIGHT] for i in range(len(contact))]
    
    ankle = inputs['ankle_angles'][0]
    # ankle_l_vel = [ankle[i + 1][LEFT] - ankle[i][LEFT] for i in left_contact]
    # ankle_r_vel = [ankle[i + 1][RIGHT] - ankle[i][RIGHT] for i in right_contact]
    ankle_l_vel = [ankle[i + 1][LEFT] - ankle[i][LEFT] for i in range(len(contact))]
    ankle_r_vel = [ankle[i + 1][RIGHT] - ankle[i][RIGHT] for i in range(len(contact))]

    print('subtalar left velocity statistics (when left foot in contact):')
    print(f'  - mean: {np.mean(subtalar_l_vel)}')
    print(f'  - sd: {np.std(subtalar_l_vel)}')
    print('subtalar right velocity statistics (when right foot in contact):')
    print(f'  - mean: {np.mean(subtalar_r_vel)}')
    print(f'  - sd: {np.std(subtalar_r_vel)}')
    
    print('mtp left velocity statistics (when left foot in contact):')
    print(f'  - mean: {np.mean(mtp_l_vel)}')
    print(f'  - sd: {np.std(mtp_l_vel)}')
    print('mtp right velocity statistics (when right foot in contact):')
    print(f'  - mean: {np.mean(mtp_r_vel)}')
    print(f'  - sd: {np.std(mtp_r_vel)}')
    
    print('ankle left velocity statistics (when left foot in contact):')
    print(f'  - mean: {np.mean(ankle_l_vel)}')
    print(f'  - sd: {np.std(ankle_l_vel)}')
    print('ankle right velocity statistics (when right foot in contact):')
    print(f'  - mean: {np.mean(ankle_r_vel)}')
    print(f'  - sd: {np.std(ankle_r_vel)}')
    
        
for i, batch in enumerate(train_dataloader):
    inputs: Dict[str, torch.Tensor]
    labels: Dict[str, torch.Tensor]
    batch_subject_indices: List[int]
    batch_trial_indices: List[int]
    inputs, labels, batch_subject_indices, batch_trial_indices = batch
    
    if i == 0:
        print(f'num input features: {len(inputs)}')
        for key, val in inputs.items():
            print(f'inputs dict tensor shape: {inputs[key].size()}')
            print(f'num inputs per batch: {len(inputs[key])}')
            print(f'num frames per input: {len(inputs[key][0])}')
            print(f'num dofs per frame: {len(inputs[key][0][0])}')
            break
    print(f'Batch {i + 1} / {len(train_dataloader)}:')
    contact = inputs['contact']
    print(f'contact: {contact}')
    process_inputs(inputs)
    print('-' * 80)
