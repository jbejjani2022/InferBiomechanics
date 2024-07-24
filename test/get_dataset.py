import os

from src.data.AddBiomechanicsDataset import AddBiomechanicsDataset
from test.FootContact import AddBiomechanicsDatasetFootContact


geometry = 'src/Geometry'
print('Using Geometry folder: ' + geometry)
geometry = os.path.abspath(geometry)
if not geometry.endswith('/'):
    geometry += '/'
    
DEV = 'test'
dataset_home = "/n/holyscratch01/pslade_lab/AddBiomechanicsDataset/addb_dataset"
train_dataset_path = os.path.abspath(os.path.join(dataset_home, 'train'))
dev_dataset_path = os.path.abspath(os.path.join(dataset_home, DEV))


def get_dataset(short, foot_contact = False, history_len = 50):
    Data = AddBiomechanicsDatasetFootContact if foot_contact else AddBiomechanicsDataset
    train_dataset = Data(train_dataset_path, history_len, geometry_folder=geometry, testing_with_short_dataset=short)
    dev_dataset = Data(dev_dataset_path, history_len, geometry_folder=geometry, testing_with_short_dataset=short)
    return train_dataset, dev_dataset
