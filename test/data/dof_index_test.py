import os

from src.data.AddBiomechanicsDataset import AddBiomechanicsDataset

geometry = 'src/Geometry'
print('Using Geometry folder: ' + geometry)
geometry = os.path.abspath(geometry)
if not geometry.endswith('/'):
    geometry += '/'
    
    
DEV = 'test'
dataset_home = "/n/holyscratch01/pslade_lab/AddBiomechanicsDataset/addb_dataset"
train_dataset_path = os.path.abspath(os.path.join(dataset_home, 'train'))
dev_dataset_path = os.path.abspath(os.path.join(dataset_home, DEV))

history_len = 50
short = False

print("TRAIN DATASET SKELETONS:")
train_dataset = AddBiomechanicsDataset(train_dataset_path, 50, geometry_folder=geometry, testing_with_short_dataset=short)
train_dataset.inspect_dof_indices()
print('-' * 80)
print("DEV DATASET SKELETONS:")
dev_dataset = AddBiomechanicsDataset(dev_dataset_path, 50, geometry_folder=geometry, testing_with_short_dataset=short)
dev_dataset.inspect_dof_indices()
