import torch
from data.AddBiomechanicsDataset import AddBiomechanicsDataset
from data.AddBiomechanicsDataset import OutputDataKeys, InputDataKeys
from typing import Dict, List, Optional
import numpy as np
import wandb
import argparse
import os
import sys

import torch.distributed as dist


components = {
    0: "left-x",
    1: "left-y",
    2: "left-z",
    3: "right-x",
    4: "right-y",
    5: "right-z"
}
wrench_components = {
    0: "left-moment-x",
    1: "left-moment-y",
    2: "left-moment-z",
    3: "left-force-x",
    4: "left-force-y",
    5: "left-force-z",
    6: "right-moment-x",
    7: "right-moment-y",
    8: "right-moment-z",
    9: "right-force-x",
    10: "right-force-y",
    11: "right-force-z"
}

class RegressionLossEvaluator:
    dataset: AddBiomechanicsDataset

    losses: List[torch.Tensor]
    pos_losses: List[torch.Tensor]
    vel_losses: List[torch.Tensor]
    acc_losses: List[torch.Tensor]
    # contact_losses: List[torch.Tensor]

    pos_reported_metrics: List[torch.Tensor]
    vel_reported_metrics: List[torch.Tensor]
    acc_reported_metrics: List[torch.Tensor]
    com_pos_reported_metrics: List[torch.Tensor]
    com_vel_reported_metrics: List[torch.Tensor]
    com_acc_reported_metrics: List[torch.Tensor]
    # contact_reported_metrics: List[torch.Tensor]

    def __init__(self, dataset: AddBiomechanicsDataset, split: str, device='cpu'):
        self.dataset = dataset
        self.split = split
        self.device = device
        self.rank = dist.get_rank()

        # Aggregating losses across batches for dev set evaluation
        self.losses = []
        self.acc_losses = []
        self.vel_losses = []
        self.pos_losses  = []
        self.com_pos_losses = []
        self.com_vel_losses = []
        self.com_acc_losses = []
        # self.contact_losses = []

        self.pos_reported_metrics = []
        self.vel_reported_metrics = []
        self.acc_reported_metrics = []
        self.com_pos_reported_metrics = []
        self.com_vel_reported_metrics = []
        self.com_acc_reported_metrics = []
        # self.contact_reported_metrics = []


    @staticmethod
    def get_squared_diff_mean_vector(output_tensor: torch.Tensor, label_tensor: torch.Tensor) -> torch.Tensor:
        if output_tensor.shape != label_tensor.shape:
            print(f'Output shape: {output_tensor.shape}\n Label Shape: {label_tensor.shape}')
            raise ValueError('Output and label tensors must have the same shape')
        if len(output_tensor.shape) != 3:
            raise ValueError('Output and label tensors must be 3-dimensional')
        if output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2] == 0:
            raise ValueError('Output and label tensors must not be empty')
        force_diff = (output_tensor - label_tensor)
        force_loss = torch.mean(force_diff ** 2, dim=(0,1))
        return force_loss
    
    @staticmethod
    def get_mean_norm_error(output_tensor: torch.Tensor, label_tensor: torch.Tensor, vec_size: int = 23) -> torch.Tensor:
        if output_tensor.shape != label_tensor.shape:
            raise ValueError('Output and label tensors must have the same shape')
        if len(output_tensor.shape) != 3:
            raise ValueError('Output and label tensors must be 3-dimensional')
        if output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2] == 0:
            raise ValueError('Output and label tensors must not be empty')
        if output_tensor.shape[-1] % vec_size != 0:
            print(output_tensor.shape[-1])
            raise ValueError('Tensors must have a final dimension divisible by vec_size=' + str(vec_size))

        diffs = output_tensor - label_tensor

        # Reshape the tensor so that the last dimension is split into chunks of `vec_size`
        reshaped_tensor = diffs.view(diffs.shape[0], diffs.shape[1], -1, vec_size)

        # Compute the norm over the last dimension
        norms = torch.norm(reshaped_tensor[:,-1:,:,:], dim=3)

        # Compute the mean norm over all the dimensions
        mean_norm = torch.mean(norms)

        return mean_norm
    
    @staticmethod
    def get_com_acc_error(output_force_tensor: torch.Tensor, label_force_tensor: torch.Tensor) -> torch.Tensor:
        if output_force_tensor.shape != label_force_tensor.shape:
            raise ValueError('Output and label tensors must have the same shape')
        if len(output_force_tensor.shape) != 3:
            raise ValueError('Output and label tensors must be 3-dimensional')
        if output_force_tensor.shape[0] * output_force_tensor.shape[1] * output_force_tensor.shape[2] == 0:
            raise ValueError('Output and label tensors must not be empty')
        if output_force_tensor.shape[-1] != 6:
            raise ValueError('Output and label tensors must have a 6 dimensional final dimension')

        # Compute the mean norm over all the dimensions
        output_force_sum = output_force_tensor[:, :, :3] + output_force_tensor[:, :, 3:]
        label_force_sum = label_force_tensor[:, :, :3] + label_force_tensor[:, :, 3:]

        return RegressionLossEvaluator.get_mean_norm_error(output_force_sum, label_force_sum, vec_size=3)
    
    def __call__(self,
                 inputs: Dict[str, torch.Tensor],
                 outputs: Dict[str, torch.Tensor],
                 labels: Dict[str, torch.Tensor],
                 batch_subject_indices: List[int],
                 batch_trial_indices: List[int],
                 args: argparse.Namespace,
                 compute_report: bool = False,
                 log_reports_to_wandb: bool = False,
                 analyze: bool = False,
                 plot_path_root: str = 'outputs/plots') -> torch.Tensor:

        ############################################################################
        # Step 1: Compute the loss
        ############################################################################

        # Perform all computations on GPU
        for key, _ in labels.items():
            labels[key] = labels[key].to(self.device)
        
        for key, _ in outputs.items():
            outputs[key] = outputs[key].to(self.device)

        # 1.1. Compute the force loss, as a single vector of length 3*N
        pos_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.POS],
            labels[OutputDataKeys.POS]
        )
        self.pos_losses.append(pos_loss)

        com_pos_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.COM_POS],
            labels[OutputDataKeys.COM_POS]
        )
        self.com_pos_losses.append(com_pos_loss)

        vel_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.VEL],
            labels[OutputDataKeys.VEL]
        )
        self.vel_losses.append(vel_loss)

        com_vel_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.COM_VEL],
            labels[OutputDataKeys.COM_VEL]
        )
        self.com_vel_losses.append(com_vel_loss)

        acc_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.ACC],
            labels[OutputDataKeys.ACC]
        )
        self.acc_losses.append(acc_loss)


        com_acc_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.COM_ACC],
            labels[OutputDataKeys.COM_ACC]
        )
        self.com_acc_losses.append(com_acc_loss)

        # contact_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
        #     outputs[OutputDataKeys.CONTACT],
        #     labels[OutputDataKeys.CONTACT][:,:,:2]
        # )
        # self.contact_losses.append(contact_loss)
        
        loss = (torch.sum(pos_loss) +
                torch.sum(com_pos_loss) +
                torch.sum(vel_loss) +
                torch.sum(com_vel_loss) +
                torch.sum(acc_loss) +
                torch.sum(com_acc_loss))
                # torch.sum(contact_loss))

        self.losses.append(loss)

        ############################################################################
        # Step 2: Compute report data if requested 
        ############################################################################

        with torch.no_grad():
            pos_reported_metric: float = RegressionLossEvaluator.get_mean_norm_error(
                outputs[OutputDataKeys.POS],
                labels[OutputDataKeys.POS]
            ).item()
            vel_reported_metric: float = RegressionLossEvaluator.get_mean_norm_error(
                outputs[OutputDataKeys.VEL],
                labels[OutputDataKeys.VEL]
            ).item()
            acc_reported_metric: float = RegressionLossEvaluator.get_mean_norm_error(
                outputs[OutputDataKeys.ACC],
                labels[OutputDataKeys.ACC]
            ).item()
            com_pos_reported_metric: float = RegressionLossEvaluator.get_mean_norm_error(
                outputs[OutputDataKeys.COM_POS],
                labels[OutputDataKeys.COM_POS],
                vec_size=3
            ).item()
            com_vel_reported_metric: float = RegressionLossEvaluator.get_mean_norm_error(
                outputs[OutputDataKeys.COM_VEL],
                labels[OutputDataKeys.COM_VEL],
                vec_size=3
            ).item()
            com_acc_reported_metric: float = RegressionLossEvaluator.get_mean_norm_error(
                outputs[OutputDataKeys.COM_ACC],
                labels[OutputDataKeys.COM_ACC],
                vec_size=3
            ).item()
            # contact_reported_metric: float = RegressionLossEvaluator.get_mean_norm_error(
            #     outputs[OutputDataKeys.CONTACT],
            #     labels[OutputDataKeys.CONTACT][:,:,:2],
            #     vec_size=2
            # ).item()

            self.pos_reported_metrics.append(pos_reported_metric)
            self.vel_reported_metrics.append(vel_reported_metric)
            self.acc_reported_metrics.append(acc_reported_metric)
            self.com_pos_reported_metrics.append(com_acc_reported_metric)
            self.com_vel_reported_metrics.append(com_vel_reported_metric)
            self.com_acc_reported_metrics.append(com_acc_reported_metric)
            # self.contact_reported_metrics.append(contact_reported_metric)

        ############################################################################
        # Step 3:  Log reports to WandB and plot results if requested
        ############################################################################

        if log_reports_to_wandb and self.rank == 0:
            self.log_to_wandb(args,
                              pos_loss,
                              vel_loss,
                              acc_loss,
                              com_pos_loss,
                              com_vel_loss,
                              com_acc_loss,
                            #   contact_loss,
                              loss,
                              pos_reported_metric,
                              vel_reported_metric,
                              acc_reported_metric,
                              com_pos_reported_metric,
                              com_vel_reported_metric,
                              com_acc_reported_metric)
                            #   contact_reported_metric)
            
        if analyze:
            self.plot_motion_error = ((outputs[OutputDataKeys.POS] - labels[OutputDataKeys.POS]) ** 2)[:, -1, :].reshape(-1, 6).detach().numpy()
            raise NotImplementedError
        
        return loss
    
    def log_to_wandb(self,
                     args: argparse.Namespace,
                     pos_loss: torch.Tensor,
                     vel_loss: torch.Tensor,
                     acc_loss: torch.Tensor,
                     com_pos_loss: torch.Tensor,
                     com_vel_loss: torch.Tensor,
                     com_acc_loss: torch.Tensor,
                    #  contact_loss: torch.Tensor
                     loss: torch.Tensor,
                     
                     pos_reported_metric: Optional[float],
                     vel_reported_metric: Optional[float],
                     acc_reported_metric: Optional[float],
                     com_pos_reported_metric: Optional[float],
                     com_vel_reported_metric: Optional[float],
                     com_acc_reported_metric: Optional[float]):
                    #  contact_reported_metric: Optional[float]):
        
        report: Dict[str, float] = {
            f'{self.split}/loss': loss.item()
        }

        if pos_reported_metric is not None:
            report[f'{self.split}/reports/Pos Avg Err'] = pos_reported_metric
        if vel_reported_metric is not None:
            report[f'{self.split}/reports/Vel Avg Err'] = vel_reported_metric
        if acc_reported_metric is not None:
            report[f'{self.split}/reports/Acc Avg Err'] = acc_reported_metric
        if com_pos_reported_metric is not None:
            report[f'{self.split}/reports/COM Pos Avg Err'] = com_pos_reported_metric
        if com_vel_reported_metric is not None:
            report[f'{self.split}/reports/COM Vel Avg Err'] = com_vel_reported_metric
        if com_acc_reported_metric is not None:
            report[f'{self.split}/reports/COM Acc Avg Err'] = com_acc_reported_metric
        # if contact_reported_metric is not None:
        #     report[f'{self.split}/reports/Contact Avg Err'] = contact_reported_metric

        wandb.log(report)

    def print_report(self,
                     args: Optional[argparse.Namespace] = None,
                     reset: bool = True,
                     log_to_wandb: bool = False):
        
        pos_reported_metric: Optional[float] = np.mean(self.pos_reported_metrics) if len(self.pos_reported_metrics) > 0 else None
        vel_reported_metric: Optional[float] = np.mean(self.vel_reported_metrics) if len(self.vel_reported_metrics) > 0 else None
        acc_reported_metric: Optional[float] = np.mean(self.acc_reported_metrics) if len(self.acc_reported_metrics) > 0 else None
        com_pos_reported_metric: Optional[float] = np.mean(self.com_pos_reported_metrics) if len(self.com_pos_reported_metrics) > 0 else None
        com_vel_reported_metric: Optional[float] = np.mean(self.com_vel_reported_metrics) if len(self.com_vel_reported_metrics) > 0 else None
        com_acc_reported_metric: Optional[float] = np.mean(self.com_acc_reported_metrics) if len(self.com_acc_reported_metrics) > 0 else None
        # contact_reported_metric: Optional[float] = np.mean(self.contact_reported_metrics) if len(self.contact_reported_metrics) > 0 else None

        if log_to_wandb and len(self.pos_losses) > 0:
            assert(args is not None)
            aggregate_pos_loss = torch.mean(torch.vstack(self.pos_losses), dim=0)
            aggregate_com_pos_loss = torch.mean(torch.vstack(self.com_pos_losses),dim=0)
            aggregate_vel_loss = torch.mean(torch.vstack(self.vel_losses), dim=0)
            aggregate_com_vel_loss = torch.mean(torch.vstack(self.com_vel_losses),dim=0)
            aggregate_acc_loss = torch.mean(torch.vstack(self.acc_losses), dim=0)
            aggregate_com_acc_loss = torch.mean(torch.vstack(self.com_acc_losses),dim=0)
            # aggregate_contact_loss = torch.mean(torch.vstack(self.contact_losses), dim=0)
            aggregate_loss = torch.mean(torch.hstack(self.losses))
            self.log_to_wandb(args,
                              aggregate_pos_loss,
                              aggregate_com_pos_loss,
                              aggregate_vel_loss,
                              aggregate_com_vel_loss,
                              aggregate_acc_loss,
                              aggregate_com_acc_loss,
                            #   aggregate_contact_loss,
                              aggregate_loss,
                              pos_reported_metric,
                              vel_reported_metric,
                              acc_reported_metric,
                              com_pos_reported_metric,
                              com_vel_reported_metric,
                              com_acc_reported_metric)
                            #   contact_reported_metric)
        if pos_reported_metric is not None:
            print(f'\tPos Avg Err: {pos_reported_metric}')
            print(f'\tVel Avg Err: {vel_reported_metric}')
            print(f'\tAcc Avg Err: {acc_reported_metric}')
            print(f'\tCOM Pos Avg Err: {com_pos_reported_metric}')
            print(f'\tCOM Vel Avg Err: {com_vel_reported_metric}')
            print(f'\tCOM Acc Avg Err: {com_acc_reported_metric}')
            # print(f'\tContact Avg Err: {contact_reported_metric}')

        if reset:
            self.losses = []
            self.pos_losses = []
            self.vel_losses = []
            self.acc_losses = []
            # self.contact_losses = []
            
            self.pos_reported_metrics = []
            self.vel_reported_metrics = []
            self.acc_reported_metrics = []
            self.com_pos_reported_metrics = []
            self.com_vel_reported_metrics = []
            self.com_acc_reported_metrics = []
            # self.contact_reported_metrics = []