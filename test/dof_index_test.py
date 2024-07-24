from test.get_dataset import get_dataset

short = False
train_dataset, dev_dataset = get_dataset(short)
print("TRAIN DATASET SKELETONS:")
train_dataset.inspect_dof_indices()
print('-' * 80)
print("DEV DATASET SKELETONS:")
dev_dataset.inspect_dof_indices()
