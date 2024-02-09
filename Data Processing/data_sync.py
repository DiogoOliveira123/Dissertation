import zipfile
import re
import pandas as pd
import csv
import os
import functions as func
import numpy as np

from os import mkdir
from tqdm import tqdm
from matplotlib import pyplot as plt

"""----------- VARIABLES ------------"""
directory_participants = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\PARTICIPANTS'
directory_labels = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\LABELS_PARTICIPANTS'
contents = os.listdir(directory_participants)
contents_labels = os.listdir(directory_labels)

NUMBER_PARTICIPANTS = 4
NUMBER_TRIALS = 24

# TIMESTAMPS VARIABLES
trials_initial_timestamp = []
rgb_timestamps = []
depth_timestamps = []
trial_sum_timestamps = []


# CSV FILES VARIABLES
trial_column_length = []
increment = float(1/60)
csv_fields_60Hz = ["Timestamp (60 Hz)"]
csv_fields_30Hz = ["Timestamp (30 Hz)"]

"""----------- ITERATE THROUGH EVERY PARTICIPANT WITH A PROGRESS BAR -----------"""
for participant in tqdm(range(len(contents)), desc="Processing Participants"):
    path_zip = os.path.join(directory_participants, contents[participant])

    """----------- OPEN ZIP FILE -----------"""
    with (zipfile.ZipFile(path_zip, 'r') as zip_ref):
        zip_contents = zip_ref.namelist()
        # All files
        folder_contents = [item for item in zip_contents if item.endswith('stamp')]

        # Filtered files
        xsens_xlsx_files = [item for item in zip_contents if item.endswith('xlsx') and item.__contains__('Xsens') and
                            item.__contains__('FC') is False]
        depth_files = [item for item in zip_contents if item.__contains__('png') and item.__contains__('gait')]
        rgb_files = [item for item in zip_contents if item.endswith('txt') and item.__contains__('gait')]

        # Sorted files (Trial 1 to 24)
        sorted_folder_contents = sorted(folder_contents, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))
        sorted_xlsx_files = sorted(xsens_xlsx_files, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))
        sorted_depth_files = sorted(depth_files, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))
        sorted_rgb_files = sorted(rgb_files, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))

        """----------- ITERATE THROUGH EVERY TRIAL WITH A PROGRESS BAR -----------"""
        for trial in tqdm(range(NUMBER_TRIALS), desc=f"Participant [{participant + 1}] - Processing Trials"):

            """----------- GET TRIALS' INITIAL TIMESTAMPS -----------"""
            trials_initial_timestamp = func.get_initial_timestamp(sorted_folder_contents[trial], '_', '.')

            """----------- GET DEPTH TIMESTAMPS -----------"""
            depth_trials = [item for item in sorted_depth_files if item.__contains__(f'Trial_{trial + 1}/')]

            depth_timestamps = func.get_depth_timestamps(depth_trials, '_', '.')

            """----------- GET RGB TIMESTAMPS -----------"""
            rgb_trials = sorted_rgb_files[trial]

            rgb_timestamps = func.get_rgb_timestamps(zip_ref, rgb_trials)

            """----------- GET SUM_FC LABELS -----------"""
            # Sum_FC Variables
            trial_labels = []
            sum_FC_labels_60Hz = []
            sum_FC_labels_30Hz = []

            path_labels_participants = os.path.join(directory_labels, contents_labels[participant])

            # Get the csv sum_FC files
            sum_FC_files = [os.path.join(root, name) for root, dirs, files in os.walk(path_labels_participants)
                            for name in files if name.endswith(".xlsx")]
            file_labels = pd.read_excel(sum_FC_files[trial])
            labels_value = file_labels.values

            # Convert it to a list
            trial_labels.append(labels_value.tolist())

            # Cycle that converts a list[list[list[]]] to a list[]
            for element in trial_labels:
                for value in element:
                    for i in value:
                        sum_FC_labels_60Hz.append(i)

            """----------- LABELS 60 HZ -> 30 HZ DOWN-SAMPLING -----------"""
            sum_FC_labels_30Hz.extend(sum_FC_labels_60Hz[::2])

            """----------- GET TRIALS' CSV FILES' LENGTHS -----------"""
            with zip_ref.open(sorted_xlsx_files[trial]) as csv_file:
                file = pd.read_excel(csv_file, 'Segment Orientation - Euler')
                column_len = file.iloc[:, 0]
                trial_column_length = (len(column_len))

            """----------- XSENS 60 HZ -> 30 HZ DOWN-SAMPLING -----------"""
            # Get trial's length and timestamp (float)
            timestamp_length = int(trial_column_length)
            timestamp_number = float(trials_initial_timestamp)

            # Initialize timestamp
            trial_timestamps = [timestamp_number]

            # Sum of timestamps
            for frame in range(1, timestamp_length):
                # Timestamps at 60 Hz
                trial_timestamps.append(timestamp_number + frame * increment)

            # Append timestamps at 30 Hz (down-sample from 60 Hz to 30 Hz)
            xsens_30hz_timestamps = []
            xsens_30hz_timestamps.extend(trial_timestamps[::2])

            """----------- MATCH TIMESTAMPS FROM XSENS WITH RGB AND DEPTH -----------"""
            # Depth Aligned Variables
            depth_timestamps_aligned = []
            position_timestamp_depth = []

            # RGB Aligned Variables
            rgb_timestamps_aligned = []
            position_timestamp_rgb = []

            # Xsens 30Hz Aligned Variables
            xsens_aligned_list = []
            position_timestamp_xsens = []

            # Get the lowest length of the 3 lists, to limit their lengths (making them all equal)
            lowest_list_length = min(len(xsens_30hz_timestamps), len(depth_timestamps),
                                     len(rgb_timestamps), len(sum_FC_labels_30Hz))

            # Get the highest initial timestamp, to define the reference for the 3 lists
            highest_timestamp = max(xsens_30hz_timestamps[0], depth_timestamps[0], rgb_timestamps[0])

            # Get the aligned lists of timestamps
            if highest_timestamp == depth_timestamps[0]:
                position_timestamp_xsens, xsens_aligned_list, position_timestamp_rgb, rgb_timestamps_aligned = func.compare_timestamps(xsens_30hz_timestamps, rgb_timestamps, depth_timestamps, lowest_list_length)
                depth_timestamps_aligned.extend(depth_timestamps[:lowest_list_length])
                # This is the referential
                for i in range(lowest_list_length):
                    position_timestamp_depth.append(i + 1)

            if highest_timestamp == rgb_timestamps[0]:
                position_timestamp_xsens, xsens_aligned_list, position_timestamp_depth, depth_timestamps_aligned = func.compare_timestamps(xsens_30hz_timestamps, depth_timestamps, rgb_timestamps, lowest_list_length)
                rgb_timestamps_aligned.extend(rgb_timestamps[:lowest_list_length])
                # This is the referential
                for i in range(lowest_list_length):
                    position_timestamp_rgb.append(i + 1)

            """----------- CREATE PLOTS FOR DATA VERIFICATION -----------"""
            # Subtract the highest initial timestamp, to decrease b in y = mx + b
            xsens_aligned_list_plot = [timestamp - highest_timestamp for timestamp in xsens_aligned_list]
            depth_timestamps_aligned_plot = [timestamp - highest_timestamp for timestamp in depth_timestamps_aligned]
            rgb_timestamps_aligned_plot = [timestamp - highest_timestamp for timestamp in rgb_timestamps_aligned]

            y = xsens_aligned_list_plot
            x_depth = depth_timestamps_aligned_plot
            x_rgb = rgb_timestamps_aligned_plot

            # Calculate the coefficients
            coefficients_depth = np.polyfit(x_depth, y, 1)
            coefficients_rgb = np.polyfit(x_rgb, y, 1)
            coefficients_depth_rgb = np.polyfit(x_rgb, x_depth, 1)

            # Print the findings
            print('m =', coefficients_depth[0])
            print('b =', coefficients_depth[1])

            print('m =', coefficients_rgb[0])
            print('b =', coefficients_rgb[1])

            print('m =', coefficients_depth_rgb[0])
            print('b =', coefficients_depth_rgb[1])

            # Get the 3 plots
            fig, axs = plt.subplots(1, 3)
            fig.suptitle(f'Timestamps Alignment Verification\nTrial: {trial + 1}, Participant: {participant + 1}')
            axs[0].plot(x_depth, y)
            axs[0].set_title(f'Timestamps - Xsens 30Hz & Depth\nm = {coefficients_depth[0]}, b = {coefficients_depth[1]}')
            axs[0].set(xlabel='Depth Timestamps', ylabel='Xsens 30Hz Timestamps')

            axs[1].plot(x_rgb, y)
            axs[1].set_title(f'Timestamps - Xsens 30Hz & RGB\nm = {coefficients_rgb[0]}, b = {coefficients_rgb[1]}')
            axs[1].set(xlabel='RGB Timestamps', ylabel='Xsens 30Hz Timestamps')

            axs[2].plot(x_rgb, x_depth)
            axs[2].set_title(f'Timestamps - Depth & RGB\nm = {coefficients_depth_rgb[0]}, b = {coefficients_depth_rgb[1]}')
            axs[2].set(xlabel='RGB Timestamps', ylabel='Depth Timestamps')

            """----------- WRITE TO CSV FILES -----------"""
            # Path for Trials folder
            participant_trial_number = os.path.join(f'Participant_{participant + 1}\\Trial_{trial + 1}',
                                                    f'Trial_{trial + 1}.csv')

            # Path for the csv file
            path_sync = os.path.join(r'C:\Users\diman\PycharmProjects\data_sync\Trials\Trials_Sync',
                                     participant_trial_number)

            mydict = [{'Frames Xsens': position_timestamp_xsens, ' Frames Depth': position_timestamp_rgb,
                       ' Frames RGB': position_timestamp_depth, ' Labels': sum_FC_labels_30Hz}
                      for position_timestamp_xsens, position_timestamp_rgb, position_timestamp_depth,
                      sum_FC_labels_30Hz
                      in zip(position_timestamp_xsens, position_timestamp_rgb, position_timestamp_depth,
                             sum_FC_labels_30Hz)]

            header = ['Frames Xsens', ' Frames Depth', ' Frames RGB', ' Labels']

            folder = f'Trial_{trial + 1}'

            # Path for the folders
            folder_trial = os.path.join(r'C:\Users\diman\PycharmProjects\data_sync\Trials\Trials_Sync',
                                        f'Participant_{participant + 1}', folder)

            # Path for the txt file
            path_txt = os.path.join(os.path.join(r'C:\Users\diman\PycharmProjects\data_sync\Trials\Trials_Sync',
                                         f'Participant_{participant + 1}', folder), 'm_and_b_values.txt')

            # Create all trials folders, where the csv, png and txt files will be
            mkdir(folder_trial)

            with open(path_sync, 'w', newline='') as csv_file_sync:
                writer = csv.DictWriter(csv_file_sync, fieldnames=header)

                # Write the header to the CSV file
                writer.writeheader()

                # Write the rows to the CSV file
                for row in mydict:
                    writer.writerow(row)

            with open(path_txt, 'w') as f:
                f.write(f'Xsens 30Hz and Depth:\nm = {coefficients_depth[0]}, b = {coefficients_depth[1]}\n')
                f.write(f'Xsens 30Hz and RGB:\nm = {coefficients_rgb[0]}, b = {coefficients_rgb[1]}\n')
                f.write(f'Depth and RGB:\nm = {coefficients_depth_rgb[0]}, b = {coefficients_depth_rgb[1]}')

            # Save the png files
            plt.savefig(os.path.join(r'C:\Users\diman\PycharmProjects\data_sync\Trials\Trials_Sync',
                                     f'Participant_{participant + 1}', folder) + r'\Sync_Plots.png')

            # Close all plots, once the trial processing is completed, to decrease the used memory when executing
            plt.close('all')
