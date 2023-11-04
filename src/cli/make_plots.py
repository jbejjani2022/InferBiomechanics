import argparse
import os
from cli.abstract_command import AbstractCommand
from typing import List, Sequence
import nimblephysics as nimble
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import seaborn as sns
import time


class MakePlotsCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('make-plots', help='Make summary plots and metrics on entire dataset.')
        subparser.add_argument('--data-path', type=str, help='Root path to all data files.')
        subparser.add_argument('--out-path', type=str, default='../figures', help='Path to output plots to.')
        subparser.add_argument('--output-histograms', action="store_true",
                               help='Whether to output summary histograms.')
        subparser.add_argument('--output-scatterplots', action="store_true",
                               help='Whether to output scatter plots.')
        subparser.add_argument('--output-errvfreq', action="store_true",
                               help='Whether to output error vs. frequency plot(s).')
        subparser.add_argument('--output-subjmetrics', action="store_true",
                               help='Whether to print subject metrics.')
        subparser.add_argument('--output-trialmetrics', action="store_true",
                               help='Whether to print trial metrics.')
        subparser.add_argument('--short', action="store_true",
                               help='Only use first few files of dataset.')
    def run(self, args: argparse.Namespace):
        """
        Iterate over all *.b3d files in a directory hierarchy,
        and generate summary plots and metrics.
        """
        if 'command' in args and args.command != 'make-plots':
            return False

        #  Make the Dataset
        dataset = Dataset(args)

        # Create and save plots and print metrics based on settings
        if dataset.output_histograms:
            dataset.plot_demographics_histograms()
            dataset.plot_demographics_by_sex_histograms()
            dataset.plot_biomechanics_metrics_histograms()
            dataset.calculate_sex_breakdown()
        if dataset.output_errvfreq:
            dataset.make_err_v_freq_plots()
        if dataset.output_scatterplots:
            dataset.make_scatter_plot_matrices()
        if dataset.output_subjmetrics:
            dataset.print_subject_metrics()
        if dataset.output_trialmetrics:
            dataset.print_trial_metrics()

        # Print how many total subjects and trials processed
        dataset.print_totals()


# # # HELPERS # # #
def plot_histograms(datas: List[Sequence], num_bins: int, colors: List[str], labels: List[str], edgecolor: str, alpha: float,
                    ylabel: str, xlabel: str, outdir: str, outname: str, plot_log_scale: bool = False):
    """
    Create a single histogram or overlaid histograms of the given input data with given plotting settings
    """
    # Check inputs
    assert (len(datas) == len(colors))

    # Find min and max of all data to be plotted for plotting settings
    combined_data = np.concatenate([np.array(data).ravel() for data in datas])
    assert (combined_data.ndim == 1)  # check to make sure flattened for taking min/max next
    datas_min = np.min(combined_data)
    datas_max = np.max(combined_data)

    bins = np.linspace(datas_min - 0.5, datas_max + 0.5, num_bins)

    plt.figure()
    for i, data in enumerate(datas):
        if len(data) == 0:  # if empty due to partitioning conditions, etc. we won't plot it
            continue

        if len(labels) != 0:
            label = labels[i]
        else:
            label = None
        plt.hist(x=data, bins=bins, color=colors[i], edgecolor=edgecolor, alpha=alpha, label=label)
        if plot_log_scale: plt.yscale("log")

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if len(labels) != 0: plt.legend(loc="best")
    plt.savefig(os.path.join(outdir, outname))
    plt.close()


# # # CLASSES # # #
class Dataset:
    """
    Load a dataset, create helpful visualizations of underlying distributions and attributes of the dataset,
    and plot analytical baselines.
    """

    def __init__(self, args: argparse.Namespace):

        # Argparse args
        data_dir: str = args.data_path
        out_dir: str = os.path.abspath(args.out_path)
        output_histograms: bool = args.output_histograms
        output_scatterplots: bool = args.output_scatterplots
        output_errvfreq: bool = args.output_errvfreq
        output_subjmetrics: bool = args.output_subjmetrics
        output_trialmetrics: bool = args.output_trialmetrics
        short: bool = args.short

        self.data_dir = data_dir
        self.out_dir = out_dir
        self.output_histograms = output_histograms
        self.output_scatterplots = output_scatterplots
        self.output_errvfreq = output_errvfreq
        self.output_subjmetrics = output_subjmetrics
        self.output_trialmetrics = output_trialmetrics
        self.short = short

        # Aggregate paths to subject data files
        self.subj_paths: List[str] = self.extract_data_files(file_type="b3d")
        if self.short:  # test on only a few subjects
            self.subj_paths = self.subj_paths[0:2]

        # Constants and processing settings
        self.num_dofs: int = 23  # hard-coded for std skel
        self.min_trial_length: int = 15  # don't process trials shorter than this TODO: put into AddB segmenting pipeline
        self.require_same_processing_passes = False  # only process trials with all the same processing passes; however, we will always require a dynamics pass and look for it.
        self.target_processing_passes = {nimble.biomechanics.ProcessingPassType.KINEMATICS,
                                         nimble.biomechanics.ProcessingPassType.LOW_PASS_FILTER,
                                         nimble.biomechanics.ProcessingPassType.DYNAMICS}  # if require_same_processing_passes is True, these are the passes we look for. technically don't need to include DYNAMICS since we explicitly check for this

        # Estimate mass for each subject (experiment)
        self.use_estimated_mass: bool = False
        self.estimated_masses: dict = {}  # init empty; only calculate if need

        # Plotting settings
        self.freqs: List[int] = [0, 3, 6, 9, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]  # for err v. freq plot
        self.verbose: bool = False  # for debugging / prototyping TODO: integrate this in

        # Only prepare data for plotting once
        self.has_prepared_data_for_plots = False

    def extract_data_files(self, file_type: str) -> List[str]:
        """
        Recursively find files of the specified data type and aggregate their full paths.
        """
        subject_paths: List[str] = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(file_type):
                    path = os.path.join(root, file)
                    subject_paths.append(path)
        return subject_paths

    def estimate_masses(self) -> dict:
        """
        Pass through the dataset to estimate mass for each subject based on F/a ratio (experiment)
        """
        print("ESTIMATING MASS...")
        estimated_mass = dict()
        for subj_path in self.subj_paths:
            subject = nimble.biomechanics.SubjectOnDisk(subj_path)
            num_trials = subject.getNumTrials()
            masses: List[float] = []
            for trial in range(num_trials):
                init_trial_length = subject.getTrialLength(trial)

                # Do a series of checks to see if okay to process this trial
                has_dynamics = False
                processing_passes = set()
                for pass_ix in range(subject.getTrialNumProcessingPasses(trial)):
                    processing_passes.add(subject.getProcessingPassType(pass_ix))
                    if subject.getProcessingPassType(pass_ix) == nimble.biomechanics.ProcessingPassType.DYNAMICS:
                        has_dynamics = True

                if self.require_same_processing_passes:
                    if processing_passes == self.target_processing_passes:
                        passes_satisfied = True
                    else:
                        passes_satisfied = False
                else:  # we don't care if the trial has all the target passes
                    passes_satisfied = True

                # We can process if these are all true:
                if has_dynamics and passes_satisfied and (init_trial_length >= self.min_trial_length):

                    frames = subject.readFrames(trial=trial, startFrame=0, numFramesToRead=init_trial_length)
                    trial_data = Trial(frames)
                    if trial_data.num_valid_frames > 0:
                        for i in range(trial_data.num_valid_frames):
                            # Estimate the mass
                            curr_mass = (np.linalg.norm(trial_data.total_grf[i, :]) /
                                         np.linalg.norm(trial_data.com_acc_kin[i, :]))
                            curr_mass = np.nan_to_num(curr_mass, nan=0.0)
                            masses.append(curr_mass)

            # Store the average and std of the estimated mass
            estimated_mass[subj_path] = (subject.getMassKg(),
                                         np.average(masses),
                                         np.std(masses))  # input mass, estimated mass, std of estims

        return estimated_mass

    def compute_err_v_freq(self, order: int, dt: float, pred: ndarray, true: ndarray) -> List[float]:
        """
        Computes RMSE between two force prediction quantities
        (i.e. finite differenced COM acc / dynamics-derived COM acc),
        from lowpass filtering at different cutoff frequencies.
        """
        # Check inputs
        assert (pred.shape == true.shape)

        errors: List[float] = []
        # Filter with each cutoff frequency and compute error
        for freq in self.freqs:
            if freq == 0:
                # We use the average of the signals to represent lowpass filtering with cutoff freq of 0 hz
                errors.append(np.sqrt(np.mean((np.mean(pred, axis=0) - np.mean(true, axis=0)) ** 2)))
            else:
                b, a = butter(N=order, Wn=freq / (0.5 * (1 / dt)), btype='low', analog=False, output='ba')
                filt_pred = np.zeros(pred.shape)
                filt_true = np.zeros(true.shape)
                for c in range(pred.shape[1]):  # filter each coordinate separately
                    filt_pred[:, c] = filtfilt(b, a, pred[:, c], padtype='constant', method='pad')
                    filt_true[:, c] = filtfilt(b, a, true[:, c], padtype='constant', method='pad')
                errors.append(np.sqrt(np.mean((filt_pred - filt_true) ** 2)))

        return errors

    def prepare_data_for_plotting(self):
        """
        Load in subject files and extract relevant data for plotting
        """
        # Only prepare the data for plotting once
        if self.has_prepared_data_for_plots:
            return
        self.has_prepared_data_for_plots = True

        # Calculate estimated mass if using
        if self.use_estimated_mass:
            self.estimated_masses = self.estimate_masses()

        if self.output_histograms:
            # INIT STORAGE
            # Subject-specific
            self.ages: List[int] = []
            self.sexes: List[int] = []  # TODO: change this back to string
            self.bmis: List[float] = []
            # Trial-specific
            self.trial_lengths: List[int] = []  # num frames
            self.forward_speeds: List[float] = []

        if self.output_scatterplots:
            # Init scatter plots matrices
            labels = ["pelvis_tilt", "pelvis_list", "pelvis_rotation",
                      "pelvis_tx", "pelvis_ty", "pelvis_tz",
                      "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
                      "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
                      "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
                      "knee_angle_l", "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
                      "lumbar_extension", "lumbar_bending", "lumbar_rotation"]  # TODO: don't hard-code; confirm that lines up to quantities correctly

            # Set up plotting color schemes
            walking_color = "blueviolet"
            running_color = "green"

            self.jointacc_vs_comacc_plots = ScatterPlotMatrix(num_rows=6, num_cols=4,
                                                         num_plots=self.num_dofs, labels=labels)
            self.jointacc_vs_totgrf_plots = ScatterPlotMatrix(num_rows=6, num_cols=4,
                                                         num_plots=self.num_dofs, labels=labels)
            self.jointacc_vs_firstcontact_plots = ScatterPlotMatrix(num_rows=6, num_cols=4,
                                                              num_plots=self.num_dofs, labels=labels)
            self.jointacc_vs_firstdist_plots = ScatterPlotMatrix(num_rows=6, num_cols=4,
                                                              num_plots=self.num_dofs, labels=labels)

            self.jointpos_vs_comacc_plots = ScatterPlotMatrix(num_rows=6, num_cols=4,
                                                         num_plots=self.num_dofs, labels=labels)
            self.jointpos_vs_totgrf_plots = ScatterPlotMatrix(num_rows=6, num_cols=4,
                                                         num_plots=self.num_dofs, labels=labels)
            self.jointpos_vs_firstcontact_plots = ScatterPlotMatrix(num_rows=6, num_cols=4,
                                                              num_plots=self.num_dofs, labels=labels)
            self.jointpos_vs_firstdist_plots = ScatterPlotMatrix(num_rows=6, num_cols=4,
                                                              num_plots=self.num_dofs, labels=labels)

            self.jointtau_vs_comacc_plots = ScatterPlotMatrix(num_rows=6, num_cols=4,
                                                         num_plots=self.num_dofs, labels=labels)
            self.jointtau_vs_totgrf_plots = ScatterPlotMatrix(num_rows=6, num_cols=4,
                                                         num_plots=self.num_dofs, labels=labels)
            self.jointtau_vs_firstcontact_plots = ScatterPlotMatrix(num_rows=6, num_cols=4,
                                                              num_plots=self.num_dofs, labels=labels)
            self.jointtau_vs_firstdist_plots = ScatterPlotMatrix(num_rows=6, num_cols=4,
                                                              num_plots=self.num_dofs, labels=labels)

            # self.comacc_vs_totgrf_x_plots = ScatterPlotMatrix(num_rows=1, num_cols=1,
            #                                                   num_plots=1, labels=labels)
            # self.comacc_vs_totgrf_y_plots = ScatterPlotMatrix(num_rows=1, num_cols=1,
            #                                                   num_plots=1, labels=labels)
            # self.comacc_vs_totgrf_z_plots = ScatterPlotMatrix(num_rows=1, num_cols=1,
            #                                                   num_plots=1, labels=labels)
            #
            # self.comacc_vs_firstcontact_plots = ScatterPlotMatrix(num_rows=1, num_cols=1,
            #                                                   num_plots=1, labels=labels)
            # self.comacc_vs_firstdist_plots = ScatterPlotMatrix(num_rows=1, num_cols=1,
            #                                                   num_plots=1, labels=labels)

        if self.output_errvfreq:
            self.acc_errs_v_freq: List[List[float]] = []
            self.grf_errs_v_freq: List[List[float]] = []
            self.tau_errs_v_freq: List[List[float]] = []  # TODO: compute this

        # Loop through each subject:
        self.num_valid_subjs = 0  # keep track of subjects we eliminate because no valid trials
        for subj_ix, subj_path in enumerate(self.subj_paths):

            print(f"Processing subject file: {subj_path}...")

            # Load the subject
            subject_on_disk = nimble.biomechanics.SubjectOnDisk(subj_path)

            # Ensure only two contact bodies. TODO: skip this whole subject for now; integrate in per-trial checking later
            if len(subject_on_disk.getGroundForceBodies()) > 2:
                print(f"Skipping {subj_path}: too many contact bodies ({len(subject_on_disk.getGroundForceBodies())})")
                continue

            # Get the subject info needed for trial processing
            num_trials = subject_on_disk.getNumTrials()
            if self.use_estimated_mass:
                mass = self.estimated_masses[subj_path][1]
            else:
                mass = subject_on_disk.getMassKg()

            if self.output_scatterplots:  # TODO: change color scheme
                color = "black"

            # Keep track of number of valid trials for this subject
            num_valid_trials = 0

            # Loop through all trials for each subject:
            for trial in range(num_trials):
                init_trial_length = subject_on_disk.getTrialLength(trial)

                # Do a series of checks to see if okay to process this trial
                has_dynamics = False
                processing_passes = set()
                for pass_ix in range(subject_on_disk.getTrialNumProcessingPasses(trial)):
                    processing_passes.add(subject_on_disk.getProcessingPassType(pass_ix))
                    if subject_on_disk.getProcessingPassType(pass_ix) == nimble.biomechanics.ProcessingPassType.DYNAMICS:
                        has_dynamics = True

                if self.require_same_processing_passes:
                    if processing_passes == self.target_processing_passes:
                        passes_satisfied = True
                    else:
                        passes_satisfied = False
                else:  # we don't care if the trial has all the target passes
                    passes_satisfied = True

                # We can process if these are all true:
                if has_dynamics and passes_satisfied and (init_trial_length >= self.min_trial_length):

                    print(f"Processing trial {trial + 1} of {num_trials}... (Subject {subj_ix+1} of {len(self.subj_paths)})")
                    frames = subject_on_disk.readFrames(trial=trial, startFrame=0, numFramesToRead=init_trial_length)
                    trial_data = Trial(frames)
                    num_valid_frames = trial_data.num_valid_frames

                    # Additional checks based on results of processing frames for given trial:
                    if num_valid_frames == 0:
                        print(f"SKIPPING TRIAL {trial + 1} due to 0 valid frames")
                        continue
                    if np.sum(trial_data.total_grf) == 0:
                        print(f"SKIPPING TRIAL {trial + 1} due to no GRF at all in the valid frames")
                        continue
                    if num_valid_frames < len(frames):
                        print(f"REMOVING SOME FRAMES on trial {trial + 1}: "
                              f"num_valid_frames: {num_valid_frames} vs. total num frames: {len(frames)}")

                    # We broke out of this iteration of trial looping if skipping trials due to reasons above;
                    # otherwise, increment number of valid trials
                    num_valid_trials += 1
                    #print(f"Current number of valid trials: {num_valid_trials}")

                    if self.output_histograms:
                        # Add to trial-specific storage:
                        self.trial_lengths.append(num_valid_frames)
                        self.forward_speeds.append(np.average(np.abs(trial_data.com_vel_dyn[:, 0])))  # avg fwd speed over trial, absolute
                        # TODO: check if first coor is fwd; add if statement for walking or running;
                        #  separate out into speeds for walking and running; maybe store tuple of value and activity label

                    if self.output_scatterplots:
                        assert (self.num_dofs == trial_data.num_joints),  f"self.num_dofs: {self.num_dofs}; trial_data.num_joints: {trial_data.num_joints}"  # check what we assume from std skel matches data
                        # joint accelerations vs. vertical component of COM acc
                        self.jointacc_vs_comacc_plots.update_plots(trial_data.com_acc_dyn[:, 1], trial_data.joint_acc_dyn,
                                                                   color)
                        # joint accelerations vs. vertical component of total GRF
                        self.jointacc_vs_totgrf_plots.update_plots(trial_data.total_grf[:, 1], trial_data.joint_acc_dyn,
                                                                   color)
                        # joint accelerations vs. contact classification of first listed contact body
                        self.jointacc_vs_firstcontact_plots.update_plots(trial_data.contact[:, 0], trial_data.joint_acc_dyn,
                                                                         color)
                        # joint accelerations vs. vertical component of GRF distribution on first listed contact body
                        self.jointacc_vs_firstdist_plots.update_plots(trial_data.grf_dist[:, 1], trial_data.joint_acc_dyn,
                                                                      color)

                        # joint positions vs. vertical component of COM acc
                        self.jointpos_vs_comacc_plots.update_plots(trial_data.com_acc_dyn[:, 1], trial_data.joint_pos_dyn,
                                                                   color)
                        # joint positions vs. vertical component of total GRF
                        self.jointpos_vs_totgrf_plots.update_plots(trial_data.total_grf[:, 1], trial_data.joint_pos_dyn,
                                                                   color)
                        # joint positions vs. contact classification of first listed contact body
                        self.jointpos_vs_firstcontact_plots.update_plots(trial_data.contact[:, 0], trial_data.joint_pos_dyn,
                                                                         color)
                        # joint positions vs. vertical component of GRF distribution on first listed contact body
                        self.jointpos_vs_firstdist_plots.update_plots(trial_data.grf_dist[:, 1], trial_data.joint_pos_dyn,
                                                                      color)

                        # joint torques vs. vertical component of COM acc
                        self.jointtau_vs_comacc_plots.update_plots(trial_data.com_acc_dyn[:, 1], trial_data.joint_tau_dyn,
                                                                   color)
                        # joint torques vs. vertical component of total GRF
                        self.jointtau_vs_totgrf_plots.update_plots(trial_data.total_grf[:, 1], trial_data.joint_tau_dyn,
                                                                   color)
                        # joint torques vs. contact classification of first listed contact body
                        self.jointtau_vs_firstcontact_plots.update_plots(trial_data.contact[:, 0], trial_data.joint_tau_dyn,
                                                                         color)
                        # joint torques vs. vertical component of GRF distribution on first listed contact body
                        self.jointtau_vs_firstdist_plots.update_plots(trial_data.grf_dist[:, 1], trial_data.joint_tau_dyn,
                                                                      color)

                        # # COM acc vs tot GRF
                        # self.comacc_vs_totgrf_x_plots.update_plots(trial_data.total_grf[:,0], trial_data.com_acc_dyn[:,0].reshape(-1,1),
                        #                                            color)
                        # self.comacc_vs_totgrf_y_plots.update_plots(trial_data.total_grf[:,1], trial_data.com_acc_dyn[:,1].reshape(-1,1),
                        #                                            color)
                        # self.comacc_vs_totgrf_z_plots.update_plots(trial_data.total_grf[:,2], trial_data.com_acc_dyn[:,2].reshape(-1,1),
                        #                                            color)
                        #
                        # # COM acc y vs contact and dist y
                        # self.comacc_vs_firstcontact_plots.update_plots(trial_data.contact[:, 0], trial_data.com_acc_dyn[:,1].reshape(-1,1),
                        #                                                color)
                        # self.comacc_vs_firstdist_plots.update_plots(trial_data.grf_dist[:, 1], trial_data.com_acc_dyn[:,1].reshape(-1,1),
                        #                                                color)

                    if self.output_errvfreq:
                        # COM acc error vs. frequency
                        acc_err_v_freq = self.compute_err_v_freq(order=2, dt=subject_on_disk.getTrialTimestep(0),
                                                          pred=trial_data.com_acc_kin,
                                                          true=trial_data.com_acc_dyn)
                        self.acc_errs_v_freq.append(acc_err_v_freq)
                        grf_err_v_freq = self.compute_err_v_freq(order=2, dt=subject_on_disk.getTrialTimestep(0),
                                                          pred=trial_data.com_acc_kin,
                                                          true=trial_data.total_grf / mass)
                        self.grf_errs_v_freq.append(grf_err_v_freq)

            # Keep tally of number of valid trials per input subject file
            print(f"FINISHED {subj_path}: num valid trials: {num_valid_trials} num trials: {num_trials}")

            # Only get demographics info and store if this subject had at least one valid trial
            if num_valid_trials >= 1:
                self.num_valid_subjs += 1
                age = subject_on_disk.getAgeYears()
                sex = subject_on_disk.getBiologicalSex()
                height = subject_on_disk.getHeightM()
                bmi = mass / (height ** 2)
                if self.output_histograms:
                    # Add to subject-specific storage
                    self.ages.append(age)
                    self.bmis.append(bmi)
                    if sex == "male":
                        sex_int = 0
                    elif sex == "female":
                        sex_int = 1
                    else:
                        sex_int = 2  # unknown
                    self.sexes.append(sex_int)

            # Convert demographics storage to arrays  # TODO: get rid of this in revamp
            if self.output_histograms:
                self.ages = np.array(self.ages)
                self.bmis = np.array(self.bmis)
                self.sexes = np.array(self.sexes)

    # TODO: add plotting for multiple models
    def plot_err_v_freq(self, errors: List[List[float]], outname: str, plot_std: bool = False):
        """
        Plots the errors vs. frequency over a single or multiple trials,
        with errors computed from filtering at different cut off frequencies
        """
        # Check inputs
        assert (len(self.freqs) > 1)
        assert (len(errors) > 1)

        errors = np.array(errors)  # make list of lists into a 2D array

        # Check that transform to array was done with correct shape
        assert (errors.shape[-1] == len(self.freqs))

        err_avg = np.average(errors, axis=0)

        # Check that averaging result has correct shape
        assert (len(err_avg) == len(self.freqs))

        plt.figure()
        plt.plot(self.freqs, err_avg)
        if plot_std:
            err_std = np.std(errors, axis=0)
            plt.fill_between(self.freqs, err_avg - err_std, err_avg + err_std, alpha=0.5)
        plt.ylabel('RMSE (N/kg)')
        plt.xlabel('cut off freq (Hz)')
        plt.savefig(os.path.join(self.out_dir, outname))

    def make_scatter_plot_matrices(self):
        """
        Save scatter plot matrices
        """
        self.prepare_data_for_plotting()

        self.jointacc_vs_comacc_plots.save_plot(self.out_dir, "jointacc_vs_comacc.png")
        self.jointacc_vs_totgrf_plots.save_plot(self.out_dir, "jointacc_vs_totgrf.png")
        self.jointacc_vs_firstcontact_plots.save_plot(self.out_dir, "jointacc_vs_firstcontact.png")
        self.jointacc_vs_firstdist_plots.save_plot(self.out_dir, "jointacc_vs_firstdist.png")

        self.jointpos_vs_comacc_plots.save_plot(self.out_dir, "jointpos_vs_comacc.png")
        self.jointpos_vs_totgrf_plots.save_plot(self.out_dir, "jointpos_vs_totgrf.png")
        self.jointpos_vs_firstcontact_plots.save_plot(self.out_dir, "jointpos_vs_firstcontact.png")
        self.jointpos_vs_firstdist_plots.save_plot(self.out_dir, "jointpos_vs_firstdist.png")

        self.jointtau_vs_comacc_plots.save_plot(self.out_dir, "jointtau_vs_comacc.png")
        self.jointtau_vs_totgrf_plots.save_plot(self.out_dir, "jointtau_vs_totgrf.png")
        self.jointtau_vs_firstcontact_plots.save_plot(self.out_dir, "jointtau_vs_firstcontact.png")
        self.jointtau_vs_firstdist_plots.save_plot(self.out_dir, "jointtau_vs_firstdist.png")

        # self.comacc_vs_totgrf_x_plots.save_plot(self.out_dir, "comacc_vs_totgrf_x.png")
        # self.comacc_vs_totgrf_y_plots.save_plot(self.out_dir, "comacc_vs_totgrf_y.png")
        # self.comacc_vs_totgrf_z_plots.save_plot(self.out_dir, "comacc_vs_totgrf_z.png")
        #
        # self.comacc_vs_firstcontact_plots.save_plot(self.out_dir, "comacc_vs_firstcontact.png")
        # self.comacc_vs_firstdist_plots.save_plot(self.out_dir, "comacc_vs_firstdist.png")

    def plot_demographics_histograms(self):
        """
        Plots histograms of demographics info over subjects
        """
        self.prepare_data_for_plotting()

        ages_to_plot = self.ages[np.where(self.ages > 0)]  # exclude unknown
        bmis_to_plot = self.bmis[np.where(self.bmis > 0)]

        plot_histograms(datas=[ages_to_plot], num_bins=6, colors=["blue"], labels=[], edgecolor="black", alpha=1,
                        ylabel="no. of subjects", xlabel="age (yrs)", outdir=self.out_dir, outname="age_histo.png")
        plot_histograms(datas=[bmis_to_plot], num_bins=6, colors=["blue"], labels=[], edgecolor="black", alpha=1,
                        ylabel="no. of subjects", xlabel="BMI (kg/m^2)", outdir=self.out_dir, outname="bmi_histo.png")
        # Calculate % of data with reported age and print out
        print(f"{np.round((len(ages_to_plot) / len(self.subj_paths)), 2) * 100}% of subjects have age info.")

    def plot_demographics_by_sex_histograms(self):
        """
        Plots histograms of demographics by sex
        """
        self.prepare_data_for_plotting()

        # Get indices for valid age and for valid sex fields
        valid_age_ix = np.where(self.ages > 0)[0]  # access within tuple return
        m_ix = np.where(self.sexes == 0)[0]  # we assign "males" to 0
        f_ix = np.where(self.sexes == 1)[0]  # we assign "females" to 1
        u_ix = np.where(self.sexes == 2)[0]  # we assign "unknown" to 2

        # Only plot if there is both age and sex
        valid_m_ix = np.intersect1d(valid_age_ix, m_ix)
        valid_f_ix = np.intersect1d(valid_age_ix, f_ix)
        valid_u_ix = np.intersect1d(valid_age_ix, u_ix)

        plot_histograms(datas=[self.ages[valid_m_ix], self.ages[valid_f_ix], self.ages[valid_u_ix]], num_bins=6, colors=["blue", "red", "yellow"], labels=["male", "female", "unknown"],
                        edgecolor="black", alpha=0.7, ylabel="no. of subjects", xlabel="age (yrs)", outdir=self.out_dir, outname="age_bysex_histo.png")
        plot_histograms(datas=[self.bmis[valid_m_ix], self.bmis[valid_f_ix], self.bmis[valid_u_ix]], num_bins=6, colors=["blue", "red", "yellow"], labels=["male", "female", "unknown"],
                        edgecolor="black", alpha=0.7, ylabel="no. of subjects", xlabel="BMI (kg/m^2)", outdir=self.out_dir, outname="bmi_bysex_histo.png")

    def plot_biomechanics_metrics_histograms(self):
        """
        Plots histograms of biomechanics metrics over all trials
        """
        self.prepare_data_for_plotting()

        plot_histograms(datas=[self.trial_lengths], num_bins=20, colors=["blue"], labels=[], edgecolor="black", alpha=1,
                        ylabel='no. of trials', xlabel='no. of frames', outdir=self.out_dir, outname='trial_length_histo.png', plot_log_scale=True)
        plot_histograms(datas=[self.forward_speeds], num_bins=20, colors=["blue"], labels=[], edgecolor="black", alpha=1,
                        ylabel='no. of trials', xlabel='absolute anterior-posterior speed (m/s)', outdir=self.out_dir, outname='speed_histo.png', plot_log_scale=True)  # TODO: separate btwn walking and running

    def make_err_v_freq_plots(self):
        """
        For accelerations, GRFs, joint torques
        """
        self.prepare_data_for_plotting()

        self.plot_err_v_freq(errors=self.acc_errs_v_freq,
                             outname='acc_err_vs_freq.png', plot_std=False)
        self.plot_err_v_freq(errors=self.grf_errs_v_freq,
                             outname='grf_err_vs_freq.png', plot_std=False)

    def calculate_sex_breakdown(self):
        """
        Calculate biological sex breakdowns
        """
        self.prepare_data_for_plotting()

        num_males = np.count_nonzero(self.sexes == 0)
        num_females = np.count_nonzero(self.sexes == 1)
        print(f"{np.round(num_males / len(self.subj_paths), 2) * 100}% of subjects are male.")
        print(f"{np.round(num_females / len(self.subj_paths), 2) * 100}% of subjects are female.")
        print("Rest of data has unknown sex.")

    def print_totals(self):  # TODO: update this because some subjects can be removed; also what happens if don't append to trial lengths
        self.prepare_data_for_plotting()

        print(f"TOTAL NUM SUBJECTS: {self.num_valid_subjs}")
        print(f"TOTAL NUM TRIALS: {len(self.trial_lengths)}")

    def print_subject_metrics(self):
        """
        For small dataset testing (to make sure aggregating data correctly)
        """
        self.prepare_data_for_plotting()

        print(f"ages: {self.ages}")
        print(f"bmis: {self.bmis}")
        print(f"sexes: {self.sexes}")

    def print_trial_metrics(self):
        """
        For small dataset testing (to make sure aggregating data correctly)
        """
        self.prepare_data_for_plotting()

        print(f"trial_lengths: {self.trial_lengths}")
        print(f"speeds: {self.forward_speeds}")


class Trial:
    """
    Combines the relevant measures across all valid frames of data for a given trial, and also store other
    trial-specific information.
    A frame is valid if it has both kinematics and dynamics processing passes completed.
    """

    def __init__(self, frames: List[nimble.biomechanics.Frame]):

        self.num_joints = len(frames[0].processingPasses[0].pos)  # get the number of joints

        # # INIT ARRAYS FOR STORAGE # #
        # From kinematics processing pass
        self.joint_pos_kin = []
        self.joint_vel_kin = []
        self.joint_acc_kin = []
        self.com_pos_kin = []
        self.com_vel_kin = []
        self.com_acc_kin = []

        # From dynamics processing pass
        self.joint_pos_dyn = []
        self.joint_vel_dyn = []
        self.joint_acc_dyn = []
        self.com_pos_dyn = []
        self.com_vel_dyn = []
        self.com_acc_dyn = []
        self.joint_tau_dyn = []

        # GRF stuff
        self.grf = []
        self.cop = []
        self.grm = []
        self.contact = []

        # Loop thru frames and extract data
        num_valid_frames = 0
        for i, frame in enumerate(frames):

            # Only use frames that are confidently not missing GRF
            not_missing_grf = 1 if frame.missingGRFReason == nimble.biomechanics.MissingGRFReason.notMissingGRF else 0

            # Check if kinematics and dynamics processing passes exist for this frame. If so, store their pass ix
            kin_processed, dyn_processed = 0, 0  # init to false
            kin_pass_ix, dyn_pass_ix = -1, -1  # init to invalid indices
            for j, processing_pass in enumerate(frame.processingPasses):
                if processing_pass.type == nimble.biomechanics.ProcessingPassType.KINEMATICS:
                    kin_processed = 1
                    kin_pass_ix = j  # store the processing pass index corresponding to the kinematics pass
                if processing_pass.type == nimble.biomechanics.ProcessingPassType.DYNAMICS:
                    dyn_processed = 1
                    dyn_pass_ix = j  # store the processing pass index corresponding to the dynamics pass

            if kin_processed and dyn_processed and not_missing_grf:  # store the data
                num_valid_frames += 1
                # From kinematics processing pass
                self.joint_pos_kin.append(frame.processingPasses[kin_pass_ix].pos)
                self.joint_vel_kin.append(frame.processingPasses[kin_pass_ix].vel)
                self.joint_acc_kin.append(frame.processingPasses[kin_pass_ix].acc)
                self.com_pos_kin.append(frame.processingPasses[kin_pass_ix].comPos)
                self.com_vel_kin.append(frame.processingPasses[kin_pass_ix].comVel)
                self.com_acc_kin.append(frame.processingPasses[kin_pass_ix].comAcc)

                # From dynamics processing pass
                self.joint_pos_dyn.append(frame.processingPasses[dyn_pass_ix].pos)
                self.joint_vel_dyn.append(frame.processingPasses[dyn_pass_ix].vel)
                self.joint_acc_dyn.append(frame.processingPasses[dyn_pass_ix].acc)
                self.com_pos_dyn.append(frame.processingPasses[dyn_pass_ix].comPos)
                self.com_vel_dyn.append(frame.processingPasses[dyn_pass_ix].comVel)
                self.com_acc_dyn.append(frame.processingPasses[dyn_pass_ix].comAcc)
                self.joint_tau_dyn.append(frame.processingPasses[dyn_pass_ix].tau)

                # GRF stuff
                self.grf.append(frame.processingPasses[dyn_pass_ix].groundContactForce)
                self.cop.append(frame.processingPasses[dyn_pass_ix].groundContactCenterOfPressure)
                self.grm.append(frame.processingPasses[dyn_pass_ix].groundContactTorque)
                self.contact.append(frame.processingPasses[dyn_pass_ix].contact)

        # Store the number of valid frames
        self.num_valid_frames = num_valid_frames
        if num_valid_frames > 0:  # only convert/store trial info if trial is valid
            #  Convert lists to arrays
            # From kinematics processing pass
            self.joint_pos_kin = np.array(self.joint_pos_kin)
            self.joint_vel_kin = np.array(self.joint_vel_kin)
            self.joint_acc_kin = np.array(self.joint_acc_kin)
            self.com_pos_kin = np.array(self.com_pos_kin)
            self.com_vel_kin = np.array(self.com_vel_kin)
            self.com_acc_kin = np.array(self.com_acc_kin)

            # From dynamics processing pass
            self.joint_pos_dyn = np.array(self.joint_pos_dyn)
            self.joint_vel_dyn = np.array(self.joint_vel_dyn)
            self.joint_acc_dyn = np.array(self.joint_acc_dyn)
            self.com_pos_dyn = np.array(self.com_pos_dyn)
            self.com_vel_dyn = np.array(self.com_vel_dyn)
            self.com_acc_dyn = np.array(self.com_acc_dyn)
            self.joint_tau_dyn = np.array(self.joint_tau_dyn)

            # GRF stuff
            self.grf = np.array(self.grf)
            self.cop = np.array(self.cop)
            self.grm = np.array(self.grm)
            self.contact = np.array(self.contact)

            # Check shapes
            assert ((self.joint_pos_kin.shape[-1] == self.num_joints) and (self.joint_pos_dyn.shape[-1] == self.num_joints)), f"{len(frames)}, {num_valid_frames}, self.joint_pos_kin.shape[-1]: {self.joint_pos_kin.shape[-1]}; self.joint_pos_dyn.shape[-1]: {self.joint_pos_dyn.shape[-1]}"
            assert ((self.joint_vel_kin.shape[-1] == self.num_joints) and (self.joint_vel_dyn.shape[-1] == self.num_joints))
            assert ((self.joint_acc_kin.shape[-1] == self.num_joints) and (self.joint_acc_dyn.shape[-1] == self.num_joints))
            assert (self.joint_tau_dyn.shape[-1] == self.num_joints)
            assert ((self.com_pos_kin.shape[-1] == 3) and (self.com_pos_dyn.shape[-1] == 3))
            assert ((self.com_vel_kin.shape[-1] == 3) and (self.com_vel_dyn.shape[-1] == 3))
            assert ((self.com_acc_kin.shape[-1] == 3) and (self.com_acc_dyn.shape[-1] == 3))
            assert (self.grf.shape[-1] == 6), f"grf shape: {self.grf.shape}"
            assert (self.cop.shape[-1] == 6)
            assert (self.grm.shape[-1] == 6)
            assert (self.contact.shape[-1] == 2)

            # Check the number of valid frames
            assert (self.num_valid_frames == self.joint_pos_kin.shape[0])  # first dim
            assert (self.num_valid_frames <= len(frames))

            # Offset gravity in y direction for COM acc
            self.com_acc_kin[:, 1] += 9.81
            self.com_acc_dyn[:, 1] += 9.81

            # Compute total GRF from both contact bodies
            self.total_grf = self.grf[:, 0:3] + self.grf[:, 3:6]

            # Compute GRF distribution (on first listed contact body; second will just be complement)
            # only if in double-support. Otherwise, 0.
            ds_mask = np.all(self.contact == 1, axis=1)  # mask for double-support
            self.grf_dist = np.zeros_like(self.total_grf)
            self.grf_dist[ds_mask, :] = np.abs(self.grf[ds_mask, 0:3]) / (np.abs(self.grf[ds_mask, 0:3]) + np.abs(self.grf[ds_mask, 3:6]) ) # dist by abs value

            # Check contact and GRF distribution
            assert (np.all(np.logical_or(self.contact == 0, self.contact == 1)))  # all contact labels either 0 or 1
            assert (np.all((self.grf_dist >= 0) & (self.grf_dist <= 1))), f"Violation found at rows: {np.where(~((self.grf_dist >= 0) & (self.grf_dist <= 1)).all(axis=1))[0]} in grf_dist: {self.grf_dist}"  # distribution must be between 0 and 1


class ScatterPlotMatrix:
    """
    Stages a matrix of scatter plots and continuously updates the plots when processing the dataset.
    """

    def __init__(self, num_rows: int, num_cols: int, num_plots: int, labels: List[str]):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_plots = num_plots
        self.labels = labels
        self.fig, self.axs = plt.subplots(num_rows, num_cols, figsize=(20, 20), constrained_layout=True)

    def update_plots(self, x: ndarray, y: ndarray, color):
        for i in range(self.num_plots):
            row = i // self.num_cols
            col = i % self.num_cols
            ax = self.axs[row, col]
            plot_title = self.labels[i]
            ax.scatter(x, y[:, i], s=0.5, alpha=0.25, color=color)
            ax.set_box_aspect(1)  # for formatting
            ax.set_title(plot_title)

    def save_plot(self, plots_outdir, outname):
        for i, ax in enumerate(self.axs.flat):
            if i >= self.num_plots:
                ax.axis("off")  # remove empty plots
        plt.tight_layout()
        self.fig.savefig(os.path.join(plots_outdir, outname))