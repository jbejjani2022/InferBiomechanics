# Make plots using cached plots value instead of having to process b3ds to save time
import argparse
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from cli.make_plots import plot_histograms, plot_activity_classification


def main(args):
    data_path = args.data_path
    out_path = args.out_path

    np.random.seed(42)
    plot_histo = True
    plot_scatter = False

    if plot_histo:
        with open(os.path.join(data_path, "activity_class.pkl"), "rb") as file:
            act_class_dict = pickle.load(file)
        with open(os.path.join(data_path, "total_trial_lengths.pkl"), "rb") as file:
            tot_trial_lengths = pickle.load(file)
        # with open(os.path.join(data_path, "grf_trial_lengths.pkl"), "rb") as file:  # not using anymore
        #     grf_trial_lengths = pickle.load(file)
        with open(os.path.join(data_path, "norm_speeds.pkl"), "rb") as file:
            norm_speeds = pickle.load(file)

        # Do some checks on the activity class dict
        if act_class_dict['walking'] != (act_class_dict['walking_overground'] + act_class_dict['walking_treadmill']):
            print(f"DIFFERENCES DETECTED IN WALKING TIMES:")
            print(f"walking time: {act_class_dict['walking']}")
            print(f"walking overground time: {act_class_dict['walking_overground']}")
            print(f"walking treadmill time: {act_class_dict['walking_treadmill']}")
        if act_class_dict['unknown'] > 0:
            print(f"Some time in 'unknown' category: {act_class_dict['unknown']}. Adding to 'other' category.")
            act_class_dict['other'] += act_class_dict['unknown']

        # Make the activity classification plot
        plot_activity_classification(act_class_dict, out_path)

        # Make the histograms
        norm_speeds = np.array(norm_speeds)
        og_norm_speeds_length = len(norm_speeds)
        norm_speeds = norm_speeds[norm_speeds <= 10]
        clipped_norm_speeds_length = len(norm_speeds)
        print(f"OG norm speeds length: {og_norm_speeds_length}")
        print(f"Clipped: {clipped_norm_speeds_length}")
        plot_histograms(datas=[norm_speeds], num_bins=6, colors=['#006BA4'], labels=[], edgecolor="black", alpha=1,
                        ylabel='no. of trials', xlabel='average speed (m/s)', outdir=out_path, outname='speed_histo.png', plot_log_scale=True)
        plot_histograms(datas=[tot_trial_lengths], num_bins=6, colors=['#006BA4'], labels=[], edgecolor="black", alpha=1,
                        ylabel='no. of trials', xlabel='no. of frames', outdir=out_path,
                        outname='trial_length_histo.png', plot_log_scale=True, manual_bins=True)


    if plot_scatter:
        with open(os.path.join(data_path, "vel_x_data.pkl"), "rb") as file:
            vel_x = pickle.load(file)
        with open(os.path.join(data_path, "vel_y_data.pkl"), "rb") as file:
            vel_y = pickle.load(file)
        with open(os.path.join(data_path, "com_x_data.pkl"), "rb") as file:
            com_x = pickle.load(file)
        with open(os.path.join(data_path, "com_y_data.pkl"), "rb") as file:
            com_y = pickle.load(file)
        with open(os.path.join(data_path, "add_r_x_data.pkl"), "rb") as file:
            add_r_x = pickle.load(file)
        with open(os.path.join(data_path, "add_r_y_data.pkl"), "rb") as file:
            add_r_y = pickle.load(file)
        with open(os.path.join(data_path, "add_l_x_data.pkl"), "rb") as file:
            add_l_x = pickle.load(file)
        with open(os.path.join(data_path, "add_l_y_data.pkl"), "rb") as file:
            add_l_y = pickle.load(file)
        with open(os.path.join(data_path, "scatter_motion_classes.pkl"), "rb") as file:
            scatter_motion_classes = pickle.load(file)

        # Scatter
        motion_settings_dict = {
            'unknown': {'color': '#FFD92F', 'marker': '|'},
            'other': {'color': '#A6CEE3', 'marker': '_'},
            'bad': {'color': '#a65628', 'marker': '^'},
            'walking': {'color': '#999999', 'marker': 'D'},
            'running': {'color': '#377eb8', 'marker': 'p'},
            'sit-to-stand': {'color': '#984ea3', 'marker': '*'},
            'stairs': {'color': '#A6CEE3', 'marker': 'H'},
            'jump': {'color': '#e41a1c', 'marker': '.'},
            'squat': {'color': '#f781bf', 'marker': 'x'},
            'lunge': {'color': '#B3DE69', 'marker': 'o'},
            'standing': {'color': '#4daf4a', 'marker': 's'},
            'transition': {'color': '#C377E0', 'marker': 'P'}
        }

        scatter_trials_target_dict = {
            'unknown': 0,
            'other': 0,
            'bad': 0,
            'walking': 20,
            'running': 20,
            'sit-to-stand': 20,
            'stairs': 20,
            'jump': 20,
            'squat': 20,
            'lunge': 0,
            'standing': 20
        }
        scatter_trials_counter_dict = {
            'unknown': 0,
            'other': 0,
            'bad': 0,
            'walking': 0,
            'running': 0,
            'sit-to-stand': 0,
            'stairs': 0,
            'jump': 0,
            'squat': 0,
            'lunge': 0,
            'standing': 0
        }
        assert (len(vel_x) == len(vel_y) == len(com_x) == len(com_y) == len(add_r_x) == len(add_r_y) == len(add_l_x) == len(add_l_y))

        plt.figure(figsize=(8, 8))
        for i in range(len(add_r_x)):
            plt.scatter(add_r_x[i], add_r_y[i], s=10, alpha=0.25, color=motion_settings_dict[scatter_motion_classes[i]]['color'], marker='.')
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.savefig(os.path.join(out_path, "add_r_scatter.png"))

        plt.figure(figsize=(8,  8))
        for i in range(len(add_l_x)):
            plt.scatter(add_l_x[i], add_l_y[i], s=10, alpha=0.25,
                        color=motion_settings_dict[scatter_motion_classes[i]]['color'], marker='.')
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.savefig(os.path.join(out_path, "add_l_scatter.png"))

        fig_com, ax_com = plt.subplots(figsize=(8, 8))
        fig_vel, ax_vel = plt.subplots(figsize=(8, 8))
        deleted_points = 0
        total_points = 0
        for i in range(len(com_x)):

            total_points += len(com_x[i])

            if (np.any(com_y[i] < -25)) or (np.any(com_y[i] > 60)):  # get rid of buggish outlier
                x_com = np.array(com_x[i])
                y_com = np.array(com_y[i])
                x_com = x_com[(y_com>-25) & (y_com<60)]
                y_com = y_com[(y_com>-25) & (y_com<60)]
                x_vel = np.array(vel_x[i])
                y_vel = np.array(vel_y[i])
                x_vel = x_vel[(y_vel > -25) & (y_vel < 60)]
                y_vel = y_vel[(y_vel > -25) & (y_vel < 60)]

                deleted_points += len(com_y[i]) - len(y_com)
            else:
                x_com = np.array(com_x[i])
                y_com = np.array(com_y[i])
                x_vel = np.array(vel_x[i])
                y_vel = np.array(vel_y[i])

            ax_com.scatter(x_com, y_com, s=10, alpha=0.25,
                        color=motion_settings_dict[scatter_motion_classes[i]]['color'], marker='.')
            ax_vel.scatter(x_vel, y_vel, s=10, alpha=0.25,
                       color=motion_settings_dict[scatter_motion_classes[i]]['color'], marker='.')
        ax_com.tick_params(axis='both', which='major', labelsize=20)
        fig_com.savefig(os.path.join(out_path, "com_all.png"))
        ax_vel.tick_params(axis='both', which='major', labelsize=20)
        fig_vel.savefig(os.path.join(out_path, "vel_all.png"))

        print(f"Deleted points: {deleted_points}")
        print(f"Total points: {total_points}")

        scatter_thresh = 0.6
        fig_com, ax_com = plt.subplots(figsize=(8, 8))
        fig_vel, ax_vel = plt.subplots(figsize=(8, 8))

        for i in range(len(com_x)):

            motion_class = scatter_motion_classes[i]
            if scatter_trials_counter_dict == scatter_trials_target_dict: break

            prob = np.random.rand()
            if (prob < scatter_thresh) and (scatter_trials_counter_dict[motion_class] < scatter_trials_target_dict[motion_class]):
                # Select the trial
                scatter_trials_counter_dict[motion_class] += 1
                #print(f"Updated scatter counter: {scatter_trials_counter_dict}")
                ax_com.scatter(com_x[i], com_y[i], s=35, alpha=0.75,
                            color=motion_settings_dict[motion_class]['color'], marker='.')
                ax_vel.scatter(vel_x[i], vel_y[i], s=35, alpha=0.75,
                               color=motion_settings_dict[motion_class]['color'], marker='.')

        ax_com.tick_params(axis='both', which='major', labelsize=20)
        ax_vel.tick_params(axis='both', which='major', labelsize=20)
        fig_com.savefig(os.path.join(out_path, "com.png"))
        fig_vel.savefig(os.path.join(out_path, "vel.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make plots from cache')
    parser.add_argument('--data-path', type=str, help='Path to cached data.', required=True)
    parser.add_argument('--out-path', type=str, help='Where to save plots to.', required=True)
    args = parser.parse_args()
    main(args)