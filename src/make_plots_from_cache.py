# Make plots using cached plots value instead of having to process b3ds to save time
import argparse
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from cli.make_plots import plot_histograms, plot_activity_classification

np.random.seed(42)

def main(args):
    data_path = args.data_path
    out_path = args.out_path

    # Load data
    # with open(os.path.join(data_path, "activity_class.pkl"), "rb") as file:
    #     act_class_dict = pickle.load(file)
    # with open(os.path.join(data_path, "total_trial_lengths.pkl"), "rb") as file:
    #     tot_trial_lengths = pickle.load(file)
    # with open(os.path.join(data_path, "grf_trial_lengths.pkl"), "rb") as file:
    #     grf_trial_lengths = pickle.load(file)

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

    # # Do some checks on the activity class dict
    # if act_class_dict['walking'] != (act_class_dict['walking_overground'] + act_class_dict['walking_treadmill']):
    #     print(f"DIFFERENCES DETECTED IN WALKING TIMES:")
    #     print(f"walking time: {act_class_dict['walking']}")
    #     print(f"walking overground time: {act_class_dict['walking_overground']}")
    #     print(f"walking treadmill time: {act_class_dict['walking_treadmill']}")
    # if act_class_dict['unknown'] > 0:
    #     print(f"Some time in 'unknown' category: {act_class_dict['unknown']}. Adding to 'other' category.")
    #     act_class_dict['other'] += act_class_dict['unknown']
    #
    # # Make the activity classification plot
    # plot_activity_classification(act_class_dict, out_path)


    # print(f"length tot: {len(tot_trial_lengths)}")
    # print(f"length grf: {len(grf_trial_lengths)}")
    # print(f"sum tot: {sum(tot_trial_lengths)}")
    # print(f"sum grf: {sum(grf_trial_lengths)}")


    # Scatter
    motion_settings_dict = {
        'unknown': {'color': '#FFD92F', 'marker': '|'},
        'other': {'color': '#A6CEE3', 'marker': '_'},
        'bad': {'color': '#a65628', 'marker': '^'},
        'walking': {'color': '#377eb8', 'marker': 'D'},
        'running': {'color': '#A6CEE3', 'marker': 'p'},
        'sit-to-stand': {'color': '#984ea3', 'marker': '*'},
        'stairs': {'color': '#999999', 'marker': 'H'},
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
        'walking': 2,
        'running': 2,
        'sit-to-stand': 2,
        'stairs': 2,
        'jump': 2,
        'squat': 2,
        'lunge': 0,
        'standing': 2
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

    plt.figure(figsize=(8, 8))
    for i in range(len(add_r_x)):
        plt.scatter(add_r_x[i], add_r_y[i], s=10, alpha=0.25, color=motion_settings_dict[scatter_motion_classes[i]]['color'], marker='.')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(os.path.join(out_path, "add_r_scatter.png"))
    plt.figure(figsize=(8,  8))
    for i in range(len(add_r_x)):
        plt.scatter(add_l_x[i], add_l_y[i], s=10, alpha=0.25,
                    color=motion_settings_dict[scatter_motion_classes[i]]['color'], marker='.')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(os.path.join(out_path, "add_l_scatter.png"))

    scatter_thresh = 0.6
    fig_com, ax_com = plt.subplots(figsize=(8, 8))
    fig_vel, ax_vel = plt.subplots(figsize=(8, 8))

    for i in range(len(com_x)):
        motion_class = scatter_motion_classes[i]
        prob = np.random.rand()
        if (prob < scatter_thresh) and (scatter_trials_counter_dict[motion_class] < scatter_trials_target_dict[motion_class]):
            # Select the trial
            scatter_trials_counter_dict[motion_class] += 1
            print(f"Updated scatter counter: {scatter_trials_counter_dict}")
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