import torch
from torch.utils.data import DataLoader
from FasterDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from models.FeedForwardRegressionBaseline import DynamicsPredictor
from models.TransformerBaseline import TransformerBaseline
from RegressionLossEvaluator import RegressionLossEvaluator
from typing import Dict, Tuple, List
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import argparse

# The window size is the number of frames we want to have as context for our model to make predictions.
window_size = 5
# The number of timesteps to skip between each frame in a given window. Data is currently all sampled at 100 Hz, so
# this means 0.2 seconds between each frame. This times window_size is the total time span of each window, which is
# currently 2.0 seconds.
stride = 20
# The batch size is the number of windows we want to load at once, for parallel training and inference on a GPU
batch_size = 1024

device = 'cpu'

# Input dofs to train on
# input_dofs = ['knee_angle_l', 'knee_angle_r', 'hip_flexion_l', 'hip_flexion_r', 'hip_adduction_l', 'hip_adduction_r']
input_dofs = ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l']

def get_model():
    # Define the model
    # hidden_size = 2 * ((len(input_dofs) * window_size * 3) + (window_size * 3))
    hidden_size = 256
    model = DynamicsPredictor(len(input_dofs), window_size, hidden_size, dropout_prob=0.0, device=device)

    return model

def get_subject_paths(data_path):
    subject_paths = []
    if os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".bin"):
                    subject_paths.append(os.path.join(root, file))
    return subject_paths

class Trainer:
    def __init__(self, args: argparse.Namespace, model: torch.nn.Module):
        self.train_losses = []
        self.train_steps = []
        self.dev_losses = []
        self.dev_steps = []
        self.global_stepper = 0
        self.epoch = 0
        self.args = args
        self.exp_name = args.exp_name

        self.model = model
        self.train_subject_paths = get_subject_paths('./data/train')
        self.dev_subject_paths = get_subject_paths('./data/dev')
        
        exp_dir = f"outputs/{args.exp_name}"
        self.model_dir_path = f"{exp_dir}/models"
        os.makedirs(self.model_dir_path, exist_ok=True)
        
        self.plot_dir_path = f"{exp_dir}/plots"
        os.makedirs(self.plot_dir_path, exist_ok=True)
        
        self.pred_dir_path = f"{exp_dir}/pred"
        os.makedirs(self.pred_dir_path, exist_ok=True)

        self.direction = {0: 'x-left', 1: 'y-left', 2: 'z-left', 3: 'x-right', 4: 'y-right', 5: 'z-right'}
    def train(self):
        # The number of epochs is the number of times we want to iterate over the entire dataset during training
        epochs = 40
        # Learning rate
        learning_rate = self.args.lr
        # learning_rate = 1e-1

        # Define the optimizer
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=learning_rate)
        for _ in range(epochs):
            self.train_epoch()

    def train_epoch(self):
        np.random.seed(self.epoch+9999)
        np.random.shuffle(self.train_subject_paths)
        
        # Iterate over the entire training dataset
        for subject_index in range(0, len(self.train_subject_paths), 20):
            train_labels = []
            train_preds = []
            
            dataset_creation = time.time()
            # Create an instance of the dataset
            train_dataset = AddBiomechanicsDataset(self.train_subject_paths[subject_index:subject_index+20], window_size, stride, input_dofs=input_dofs, device=torch.device(device))
            # Create a DataLoader to load the data in batches
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            dataset_creation = time.time() - dataset_creation
            logging.info(f"{dataset_creation=}")
            data_start = time.time()
            for i, batch in enumerate(train_dataloader):
                loss_evaluator = RegressionLossEvaluator(contact_forces_weight=1.)
                data_time = time.time() - data_start

                forward_pass = time.time()
                inputs: Dict[str, torch.Tensor]
                labels: Dict[str, torch.Tensor]
                inputs, labels = batch

                # Clear the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                forward_pass = time.time() - forward_pass
                
                force_labels = labels[OutputDataKeys.CONTACT_FORCES].numpy()
                train_preds.append(outputs[OutputDataKeys.CONTACT_FORCES].detach().numpy())
                train_labels.append(force_labels)
                
                logging.info(f"Label stats: {np.max(force_labels, axis=0)=}, {np.min(force_labels, axis=0)=}, {np.mean(np.abs(force_labels), axis=0)=}, {np.mean(force_labels, axis=0)=}")
                # Compute the loss
                backprop = time.time()
                loss = loss_evaluator(outputs, labels)
                
                self.train_losses.append(np.sqrt(loss_evaluator.sum_contact_forces_N_error / loss_evaluator.sum_timesteps))
                self.train_steps.append(self.global_stepper)
                
                if i % 100 == 0:
                    logging.info(f'  - Batch {subject_index} / {len(self.train_subject_paths)}')
                if i % 100 == 0:
                    loss_evaluator.print_report()
                    model_path = os.path.join(self.model_dir_path, f"epoch_{self.epoch}_batch_{i}.pt")
                    torch.save({
                    'args': args,
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, model_path)
                    
                    self.plot_losses()
                    self.plot_preds(train_labels, train_preds, split="train")

                # Backward pass
                loss.backward()

                # Update the model's parameters
                self.optimizer.step()
                backprop = time.time() - backprop
                logging.info(f"{data_time=}, {forward_pass=}, {backprop=}")
                data_start = time.time()
                self.global_stepper += 1
            # # Report training loss on this epoch
            # print('Epoch '+str(epoch)+': ')
            # print('Training Set Evaluation: ')
            # loss_evaluator.print_report()
            
            # At the end of each epoch, evaluate the model on the dev set
            dev_loss_evaluator = RegressionLossEvaluator(contact_forces_weight=1.0)

            for subject_index in range(0, len(self.dev_subject_paths), 20):
                dev_dataset = AddBiomechanicsDataset(self.dev_subject_paths[subject_index:subject_index+20], window_size, stride, input_dofs=input_dofs, device=torch.device(device))
                dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
                
                with torch.no_grad():
                    for i, batch in enumerate(dev_dataloader):
                        if i % 100 == 0:
                            logging.info(f'  - Dev Batch {subject_index} / {len(self.dev_subject_paths)}')
                        inputs: Dict[str, torch.Tensor]
                        labels: Dict[str, torch.Tensor]
                        inputs, labels = batch
                        outputs = self.model(inputs)
                        loss = dev_loss_evaluator(outputs, labels)
                        # print(f"{labels[OutputDataKeys.CONTACT_FORCES].shape=}, {outputs[OutputDataKeys.CONTACT_FORCES].shape=}")
                        self.plot_preds([labels[OutputDataKeys.CONTACT_FORCES].numpy()], [outputs[OutputDataKeys.CONTACT_FORCES].numpy()], split="dev")
            self.dev_losses.append(np.sqrt(dev_loss_evaluator.sum_contact_forces_N_error / dev_loss_evaluator.sum_timesteps))
            self.dev_steps.append(self.global_stepper)
            # Report dev loss on this epoch
            logging.info('Dev Set Evaluation: ')
            dev_loss_evaluator.print_report()
        self.epoch += 1
    
    def plot_losses(self):  
        train_losses = np.concatenate(self.train_losses)
        print(f"{train_losses.shape=}")
        dev_losses = None if not self.dev_losses else np.concatenate(self.dev_losses)

        for i in range(6):
            plt.clf()
            plt.plot(self.train_steps, train_losses[:,i], label='train')
            if dev_losses is not None:
                plt.plot(self.dev_steps, dev_losses[:,i], label='dev')
            plt.legend()
            plt.savefig(os.path.join(self.plot_dir_path, f"loss-{self.direction[i]}.png"))

    def plot_preds(self, labels, preds, split="train"):
        for i in range(6):
            plt.clf()
            plt.plot(np.concatenate(labels)[::5,i], label=f'True F{self.direction[i]}')
            plt.plot(np.concatenate(preds)[::5,i], label=f'Pred F{self.direction[i]}')
            plt.legend()
            plt.savefig(os.path.join(self.pred_dir_path, f"{split}-f{self.direction[i]}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="run1")
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    exp_dir = f"outputs/{args.exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    logpath = os.path.join(exp_dir, "log")
    # Create and configure logger
    logging.basicConfig(filename=logpath,
                        format='%(asctime)s %(message)s',
                        filemode='a')

    # Creating an object
    logger = logging.getLogger()
    
    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.INFO)

    model = get_model()

    trainer = Trainer(args, model)
    trainer.train()
