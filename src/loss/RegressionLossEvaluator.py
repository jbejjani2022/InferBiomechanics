import torch
import torch.distributed as dist
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, OutputDataKeys, InputDataKeys
from typing import Dict, List, Optional
import numpy as np
import wandb
import logging
import matplotlib.pyplot as plt
import os
import argparse


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
    force_losses: List[torch.Tensor]
    moment_losses: List[torch.Tensor]
    wrench_losses: List[torch.Tensor]
    cop_losses: List[torch.Tensor]

    force_reported_metrics: List[float]
    moment_reported_metrics: List[float]
    cop_reported_metrics: List[float]
    wrench_reported_metrics: List[float]
    tau_reported_metrics: List[float]
    com_acc_reported_metrics: List[float]

    def __init__(self, dataset: AddBiomechanicsDataset, split: str):
        self.dataset = dataset
        self.split = split

        # Aggregating losses across batches for dev set evaluation
        self.losses = []
        self.force_losses = []
        self.moment_losses = []
        self.wrench_losses = []
        self.cop_losses = []

        # Aggregating reported metrics for dev set evaluation
        self.force_reported_metrics = []
        self.moment_reported_metrics = []
        self.cop_reported_metrics = []
        self.wrench_reported_metrics = []
        self.wrench_moment_reported_metrics = []
        self.tau_reported_metrics = []
        self.com_acc_reported_metrics = []
        
        # Get device
        self.rank = dist.get_rank()

    @staticmethod
    def get_squared_diff_mean_vector(output_tensor: torch.Tensor, label_tensor: torch.Tensor) -> torch.Tensor:
        # print(f'Label tensor: {label_tensor.shape}\n Output tensor: {output_tensor.shape}')
        if output_tensor.shape != label_tensor.shape:
            raise ValueError('Output and label tensors must have the same shape')
        if len(output_tensor.shape) != 3:
            raise ValueError('Output and label tensors must be 3-dimensional')
        if output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2] == 0:
            raise ValueError('Output and label tensors must not be empty')
        force_diff = (output_tensor - label_tensor)
        force_loss = torch.mean(force_diff ** 2, dim=(0,1))
        return force_loss

    @staticmethod
    def get_mask_by_threes(tensor: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        with torch.no_grad():
            if len(tensor.shape) != 3:
                raise ValueError('Mask tensor must be 3-dimensional')
            if tensor.shape[0] * tensor.shape[1] * tensor.shape[2] == 0:
                raise ValueError('Mask tensor must not be empty')
            if tensor.shape[-1] % 3 != 0:
                raise ValueError('Mask tensor must have a final dimension divisible by 3')

            # Reshape the tensor so that the last dimension is split into chunks of 3
            reshaped_tensor = tensor.view(tensor.shape[0], tensor.shape[1], -1, 3)

            # Compute the norm over the last dimension
            norms = torch.norm(reshaped_tensor, dim=-1)

            # Create a mask where the norm is greater than the threshold
            mask = (norms > threshold).float()

            # Expand the mask to cover the original last dimension size
            expanded_mask = mask.unsqueeze(3).expand(-1, -1, -1, 3)
            # print(f"{expanded_mask.shape=}")
            # Reshape the expanded mask back to the original tensor shape
            return expanded_mask.reshape(tensor.shape)

            # Un-vectorized version:

            # mask_tensor = torch.zeros_like(tensor)
            # for batch in range(mask_tensor.shape[0]):
            #     for timestep in range(mask_tensor.shape[1]):
            #         for i in range(mask_tensor.shape[2] // 3):
            #             mask_tensor[batch, timestep, i * 3:(i + 1) * 3] = (torch.norm(tensor[batch, timestep, i * 3:(i + 1) * 3]) > threshold).float()
            # return mask_tensor

    @staticmethod
    def get_mean_norm_error(output_tensor: torch.Tensor, label_tensor: torch.Tensor, vec_size: int = 3) -> torch.Tensor:
        if output_tensor.shape != label_tensor.shape:
            raise ValueError('Output and label tensors must have the same shape')
        if len(output_tensor.shape) != 3:
            raise ValueError('Output and label tensors must be 3-dimensional')
        if output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2] == 0:
            raise ValueError('Output and label tensors must not be empty')
        if output_tensor.shape[-1] % vec_size != 0:
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
        
        # 1.1. Compute the force loss, as a single vector of length 3*N
        force_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME],
            labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME]
        )
        self.force_losses.append(force_loss)
        # 1.2. Compute the moment loss, as a single vector of length 3*N
        moment_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME],
            labels[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME]
        )
        self.moment_losses.append(moment_loss)
        # 1.3. Compute the wrench loss, as a single vector of length 6*N. Note that this is NOT MASKED the same way the
        # CoP loss is, because unlike the CoP the forces and moments in the wrench are well behaved when the foot is not
        # in contact with the ground: they simply go to zero.
        wrench_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME],
            labels[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME]
        )
        self.wrench_losses.append(wrench_loss)
        # 1.4. CoP loss is tricky, because when there is no force the CoP is meaningless, and so we want to ensure that
        # we only report CoP loss on the frames where there is a 10N force or more.
        with torch.no_grad():
            cop_mask_tensor = RegressionLossEvaluator.get_mask_by_threes(
                labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME],
                threshold=10.0
            )
        cop_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME] * cop_mask_tensor,
            labels[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME] * cop_mask_tensor
        )
        self.cop_losses.append(cop_loss)
        # 1.5. We allow the user to specify which axis of the force, moment, and wrench vectors to use for computing
        # the loss. We do this by indexing into the loss vectors with the specified indices, and taking the sums.
        loss = (torch.sum(force_loss[args.predict_grf_components]) +
                torch.sum(cop_loss[args.predict_cop_components]) +
                torch.sum(moment_loss[args.predict_moment_components]) +
                torch.sum(wrench_loss[args.predict_wrench_components]))
        self.losses.append(loss)

        ############################################################################
        # Step 2: Compute report data, if we are asked to do so
        ############################################################################

        # 2.1. Initialize paper-reported values we will send to wandb, if requested
        tau_reported_metric: Optional[float] = None

        with torch.no_grad():
            # 2.2. Compute the norm errors for the force, moment, CoP, and wrench vectors
            force_reported_metric: float = RegressionLossEvaluator.get_mean_norm_error(
                outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME],
                labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME]
            ).item()
            moment_reported_metric: float = RegressionLossEvaluator.get_mean_norm_error(
                outputs[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME],
                labels[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME]
            ).item()
            cop_reported_metric: float = RegressionLossEvaluator.get_mean_norm_error(
                outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME] * cop_mask_tensor,
                labels[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME] * cop_mask_tensor
            ).item()
            wrench_moment_reported_metric_1: float = RegressionLossEvaluator.get_mean_norm_error(
                outputs[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME][:, :, :3],
                labels[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME][:, :, :3],
                vec_size=3
            ).item()
            wrench_moment_reported_metric_2: float = RegressionLossEvaluator.get_mean_norm_error(
                outputs[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME][:, :, 6:9],
                labels[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME][:, :, 6:9],
                vec_size=3
            ).item()
            wrench_moment_reported_metric: float = (wrench_moment_reported_metric_1 + wrench_moment_reported_metric_2) / 2.0
            wrench_reported_metric: float = RegressionLossEvaluator.get_mean_norm_error(
                outputs[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME],
                labels[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME],
                vec_size=6
            ).item()
            com_acc_reported_metric: float = RegressionLossEvaluator.get_com_acc_error(
                outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME],
                labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME]
            ).item()

            if compute_report:
                # 2.3. Manually compute the inverse dynamics torque errors frame-by-frame
                num_batches = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].shape[0]
                tau_reported_metric = 0.0
                num_batches = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].shape[0]
                for batch in range(num_batches):
                    skel = self.dataset.skeletons[batch_subject_indices[batch]]
                    skel.setPositions(inputs[InputDataKeys.POS][batch, -1, :].cpu().numpy())
                    skel.setVelocities(inputs[InputDataKeys.VEL][batch, -1, :].cpu().numpy())
                    acc = inputs[InputDataKeys.ACC][batch, -1, :].cpu().numpy()
                    contact_bodies = self.dataset.skeletons_contact_bodies[batch_subject_indices[batch]]
                    contact_wrench_guesses = outputs[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME][batch,
                                                -1, :].cpu().numpy() * skel.getMass()
                    contact_wrench_guesses_list = [contact_wrench_guesses[i * 6:i * 6 + 6] for i in
                                                    range(len(contact_bodies))]
                    tau = skel.getInverseDynamicsFromPredictions(acc, contact_bodies, contact_wrench_guesses_list,
                                                                    np.zeros(6))
                    tau_error = tau - labels[OutputDataKeys.TAU][batch, -1, :].cpu().numpy()
                    # Exclude root residual from error
                    tau_reported_metric += np.mean(np.abs(tau_error[6:])) / skel.getMass()
                tau_reported_metric /= num_batches
                self.tau_reported_metrics.append(tau_reported_metric)
            # 2.4. Keep track of the reported metrics for reporting averages across the entire dev set
            self.force_reported_metrics.append(force_reported_metric)
            self.moment_reported_metrics.append(moment_reported_metric)
            self.cop_reported_metrics.append(cop_reported_metric)
            self.wrench_reported_metrics.append(wrench_reported_metric)
            self.wrench_moment_reported_metrics.append(wrench_moment_reported_metric)
            self.com_acc_reported_metrics.append(com_acc_reported_metric)

        ############################################################################
        # Step 3: Log reports to wandb and plot results, if requested
        ############################################################################

        # 3.1. If requested, log the reports to Weights and Biases
        if log_reports_to_wandb and self.rank == 0:
            self.log_to_wandb(args,
                              force_loss,
                              cop_loss,
                              moment_loss,
                              wrench_loss,
                              loss,
                              force_reported_metric,
                              cop_reported_metric,
                              moment_reported_metric,
                              com_acc_reported_metric,
                              wrench_reported_metric,
                              tau_reported_metric)

        # 3.2. If requested, plot the results
        if analyze:
            self.plot_ferror = ((outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME] - labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME]) ** 2)[:, -1, :].reshape(-1, 6).detach().numpy()
            for i in args.predict_grf_components:
                plt.clf()
                plt.plot(self.plot_ferror[:, i])
                plt.savefig(os.path.join(plot_path_root,
                                         f"{os.path.basename(self.dataset.subject_paths[batch_subject_indices[0]])}_{self.dataset.subjects[batch_subject_indices[0]].getTrialName(batch_trial_indices[0])}_grferror{components[i]}.png"))
        return loss

    def log_to_wandb(self,
                     args: argparse.Namespace,
                     force_loss: torch.Tensor,
                     cop_loss: torch.Tensor,
                     moment_loss: torch.Tensor,
                     wrench_loss: torch.Tensor,
                     loss: torch.Tensor,
                     # IMPORTANT: THESE ARE NOT THE SAME AS THE LOSS VALUES ABOVE! These compute errors per vector pair
                     # as a simple norm, and take the mean of that. The above losses are squared errors, and are summed.
                     # If we were to then square-root the above losses, we would get higher values than the ones
                     # reported here:
                     force_reported_metric: Optional[float],
                     cop_reported_metric: Optional[float],
                     moment_reported_metric: Optional[float],
                     com_acc_reported_metric: Optional[float],
                     wrench_reported_metric: Optional[float],
                     tau_reported_metric: Optional[float]):

        report: Dict[str, float] = {
            **{f'{self.split}/force_rmse/{components[i]}': force_loss[i].item() ** 0.5 for i in
               args.predict_grf_components},
            **{f'{self.split}/cop_rmse/{components[i]}': cop_loss[i].item() ** 0.5 for i in
               args.predict_cop_components},
            **{f'{self.split}/moment_rmse/{components[i]}': moment_loss[i].item() ** 0.5 for i in
               args.predict_moment_components},
            **{f'{self.split}/wrench_loss/{wrench_components[i]}': wrench_loss[i].item() ** 0.5 for i in
               args.predict_wrench_components},
            f'{self.split}/loss': loss.item()
        }
        if force_reported_metric is not None:
            report[f'{self.split}/reports/Force Avg Err (N per kg)'] = force_reported_metric
        if com_acc_reported_metric is not None:
            report[f'{self.split}/reports/CoP Avg Err (m)'] = cop_reported_metric
        if moment_reported_metric is not None:
            report[f'{self.split}/reports/Moment Avg Err (Nm per kg)'] = moment_reported_metric
        if wrench_reported_metric is not None:
            report[f'{self.split}/reports/COM Acc Avg Err (m per s^2)'] = com_acc_reported_metric
        if wrench_reported_metric is not None:
            report[f'{self.split}/reports/Wrench Avg Err (N+Nm per kg)'] = wrench_reported_metric
        if tau_reported_metric is not None:
            report[f'{self.split}/reports/Non-root Joint Torques (Inverse Dynamics) Avg Err (Nm per kg)'] = tau_reported_metric

        wandb.log(report)

    def print_report(self,
                     args: Optional[argparse.Namespace] = None,
                     reset: bool = True,
                     log_to_wandb: bool = False):

        force_reported_metric: Optional[float] = np.mean(self.force_reported_metrics) if len(self.force_reported_metrics) > 0 else None
        moment_reported_metric: Optional[float] = np.mean(self.moment_reported_metrics) if len(self.moment_reported_metrics) > 0 else None
        cop_reported_metric: Optional[float] = np.mean(self.cop_reported_metrics) if len(self.cop_reported_metrics) > 0 else None
        wrench_reported_metric: Optional[float] = np.mean(self.wrench_reported_metrics) if len(self.wrench_reported_metrics) > 0 else None
        wrench_moment_reported_metric: Optional[float] = np.mean(self.wrench_moment_reported_metrics) if len(self.wrench_moment_reported_metrics) > 0 else None
        tau_reported_metric: Optional[float] = np.mean(self.tau_reported_metrics) if len(self.tau_reported_metrics) > 0 else None
        com_acc_reported_metric: Optional[float] = np.mean(self.com_acc_reported_metrics) if len(self.com_acc_reported_metrics) > 0 else None

        if log_to_wandb and len(self.force_losses) > 0:
            assert(args is not None)
            aggregate_force_loss = torch.mean(torch.vstack(self.force_losses), dim=0)
            aggregate_cop_loss = torch.mean(torch.vstack(self.cop_losses), dim=0)
            aggregate_moment_loss = torch.mean(torch.vstack(self.moment_losses), dim=0)
            aggregate_wrench_loss = torch.mean(torch.vstack(self.wrench_losses), dim=0)
            aggregate_loss = torch.mean(torch.hstack(self.losses))
            self.log_to_wandb(args,
                              aggregate_force_loss,
                              aggregate_cop_loss,
                              aggregate_moment_loss,
                              aggregate_wrench_loss,
                              aggregate_loss,
                              force_reported_metric,
                              cop_reported_metric,
                              moment_reported_metric,
                              com_acc_reported_metric,
                              wrench_reported_metric,
                              tau_reported_metric)

        if force_reported_metric is not None and self.rank == 0:
            print(f'\tForce Avg Err: {force_reported_metric} N / kg')
            print(f'\tCOM Acc Avg Err: {com_acc_reported_metric} m / s^2')
            print(f'\tCoP Avg Err: {cop_reported_metric} m')
            print(f'\tMoment Avg Err: {moment_reported_metric} Nm / kg')
            print(f'\tWrench Avg Err: {wrench_reported_metric} N+Nm / kg')
            print(f'\tWrench Moment Avg Err: {wrench_moment_reported_metric} Nm / kg')
            print(
                f'\tNon-root Joint Torques (Inverse Dynamics) Avg Err: {tau_reported_metric} Nm / kg')

        # Reset
        if reset:
            # Aggregating losses across batches for dev set evaluation
            self.losses = []
            self.force_losses = []
            self.moment_losses = []
            self.wrench_losses = []
            self.cop_losses = []

            # Aggregating reported metrics for dev set evaluation
            self.force_reported_metrics = []
            self.moment_reported_metrics = []
            self.cop_reported_metrics = []
            self.wrench_reported_metrics = []
            self.tau_reported_metrics = []
            self.com_acc_reported_metrics = []
