import zipfile
import re
import pandas as pd
import csv
import os
import functions as func
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt

"""----------- VARIABLES ------------"""
directory_participants = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\PARTICIPANTS'
directory_labels = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\LABELS_PARTICIPANTS_v2'
contents = os.listdir(directory_participants)
contents_labels = os.listdir(directory_labels)
sorted_contents_labels = sorted(contents_labels, key=lambda x: int(re.search(r'Participant_(\d+)', x).group(1)))

NUMBER_PARTICIPANTS = 1
NUMBER_TRIALS = 3

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

# ClASSES COUNT VARIABLES
total_num_ones = 0
total_num_n_ones = 0
total_num_zeros = 0

print("For data synchronization select: 1\n"
      "For video to frames extraction select: 2\n"
      "For frame labeling, by typing into image select: 3\n")

user_input = input("Select one of the options above: ")

match user_input:
    case '1':
        """----------- DATA SYNCHRONIZATION -----------"""
        """----------- ITERATE THROUGH EVERY PARTICIPANT WITH A PROGRESS BAR -----------"""
        for participant in tqdm(range(NUMBER_PARTICIPANTS), desc="Processing Participants"):
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
                video_files = [item for item in zip_contents if item.endswith('avi') and item.__contains__('gait')]

                # Sorted files (Trial 1 to 24)
                sorted_folder_contents = sorted(folder_contents, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))
                sorted_xlsx_files = sorted(xsens_xlsx_files, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))
                sorted_depth_files = sorted(depth_files, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))
                sorted_rgb_files = sorted(rgb_files, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))
                sorted_video_files = sorted(video_files, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))

                """----------- ITERATE THROUGH EVERY TRIAL WITH A PROGRESS BAR -----------"""
                for trial in tqdm(range(1, NUMBER_TRIALS), desc=f"Participant [{participant + 1}] - Processing Trials"):

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

                    path_labels_participants = os.path.join(directory_labels, sorted_contents_labels[participant])

                    # Get the csv sum_FC files
                    sum_FC_files = [os.path.join(root, name) for root, dirs, files in os.walk(path_labels_participants)
                                    for name in files if name.endswith(".csv")]
                    sorted_sum_FC_files = sorted(sum_FC_files, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))

                    file_labels = pd.read_csv(sorted_sum_FC_files[trial], header=None)
                    labels_value = file_labels.values

                    # Convert it to a list
                    trial_labels.append(labels_value.tolist())

                    # Loop that converts a list[list[list[]]] to a list[]
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

                    sum_FC_labels_30Hz_aligned = []

                    # Get the lowest length of the 3 lists, to limit their lengths (making them all equal)
                    lowest_list_length = min(len(xsens_30hz_timestamps), len(depth_timestamps),
                                             len(rgb_timestamps), len(sum_FC_labels_30Hz))

                    if not len(xsens_30hz_timestamps) == len(sum_FC_labels_30Hz):
                        raise Exception('Xsens Timestamps must have the same length as the Labels.')

                    # Get the highest initial timestamp, to define the reference for the 3 lists
                    highest_timestamp = max(xsens_30hz_timestamps[0], depth_timestamps[0], rgb_timestamps[0])

                    # Get the aligned lists of timestamps
                    if highest_timestamp == depth_timestamps[0]:
                        position_timestamp_xsens, xsens_aligned_list, position_timestamp_rgb, rgb_timestamps_aligned = func.compare_timestamps(xsens_30hz_timestamps, rgb_timestamps, depth_timestamps, lowest_list_length)
                        depth_timestamps_aligned.extend(depth_timestamps[:lowest_list_length])
                        # This is the referential
                        for i in range(lowest_list_length):
                            position_timestamp_depth.append(i)

                    if highest_timestamp == rgb_timestamps[0]:
                        position_timestamp_xsens, xsens_aligned_list, position_timestamp_depth, depth_timestamps_aligned = func.compare_timestamps(xsens_30hz_timestamps, depth_timestamps, rgb_timestamps, lowest_list_length)
                        rgb_timestamps_aligned.extend(rgb_timestamps[:lowest_list_length])
                        # This is the referential
                        for i in range(lowest_list_length):
                            position_timestamp_rgb.append(i)

                    if highest_timestamp == xsens_30hz_timestamps[0]:
                        position_timestamp_rgb, rgb_timestamps_aligned, position_timestamp_depth, depth_timestamps_aligned = func.compare_timestamps(rgb_timestamps, depth_timestamps, xsens_30hz_timestamps, lowest_list_length)
                        xsens_aligned_list.extend(xsens_30hz_timestamps[:lowest_list_length])
                        # This is the referential
                        for i in range(lowest_list_length):
                            position_timestamp_xsens.append(i + 1)

                    for i in range(lowest_list_length):
                        sum_FC_labels_30Hz_aligned.append(sum_FC_labels_30Hz[position_timestamp_xsens[i]])

                    # Get number of '1', '-1' and '0'
                    num_ones = sum_FC_labels_30Hz_aligned.count(1)
                    num_n_ones = sum_FC_labels_30Hz_aligned.count(-1)
                    num_zeros = sum_FC_labels_30Hz_aligned.count(0)

                    total_num_ones += num_ones
                    total_num_n_ones += num_n_ones
                    total_num_zeros += num_zeros

                    """----------- CREATE PLOTS FOR DATA VERIFICATION -----------"""
                    # Subtract the highest initial timestamp, to decrease b in y = mx + b
                    xsens_aligned_list_plot = [timestamp - highest_timestamp for timestamp in
                                               xsens_aligned_list]
                    depth_timestamps_aligned_plot = [timestamp - highest_timestamp for timestamp in
                                                     depth_timestamps_aligned]
                    rgb_timestamps_aligned_plot = [timestamp - highest_timestamp for timestamp in
                                                   rgb_timestamps_aligned]

                    y = xsens_aligned_list_plot
                    x_depth = depth_timestamps_aligned_plot
                    x_rgb = rgb_timestamps_aligned_plot

                    # Calculate the coefficients
                    coefficients_depth = np.polyfit(x_depth, y, 1)
                    coefficients_rgb = np.polyfit(x_rgb, y, 1)
                    coefficients_depth_rgb = np.polyfit(x_rgb, x_depth, 1)

                    # Get the 3 plots
                    fig, axs = plt.subplots(1, 3)
                    fig.suptitle(f'Timestamps Alignment Verification\nTrial: {trial + 1},'
                                 f' Participant: {participant + 1}')
                    axs[0].plot(x_depth, y)
                    axs[0].set_title(f'Timestamps - Xsens 30Hz & Depth\nm = {coefficients_depth[0]},'
                                     f' b = {coefficients_depth[1]}')
                    axs[0].set(xlabel='Depth Timestamps', ylabel='Xsens 30Hz Timestamps')

                    axs[1].plot(x_rgb, y)
                    axs[1].set_title(f'Timestamps - Xsens 30Hz & RGB\nm = {coefficients_rgb[0]},'
                                     f' b = {coefficients_rgb[1]}')
                    axs[1].set(xlabel='RGB Timestamps', ylabel='Xsens 30Hz Timestamps')

                    axs[2].plot(x_rgb, x_depth)
                    axs[2].set_title(f'Timestamps - Depth & RGB\nm = {coefficients_depth_rgb[0]},'
                                     f' b = {coefficients_depth_rgb[1]}')
                    axs[2].set(xlabel='RGB Timestamps', ylabel='Depth Timestamps')

                    """----------- WRITE TO CSV FILES -----------"""
                    # Path for Trials folder
                    participant_trial_number = os.path.join(f'Participant_{participant + 1}\\Trial_{trial + 1}',
                                                            f'Trial_{trial + 1}.csv')

                    # Path for the csv file
                    path_sync = os.path.join(r'C:\Users\diman\PycharmProjects\dissertation\Trials\Trials_Sync_v2',
                                             participant_trial_number)

                    mydict = [{'Position Xsens': position_timestamp_xsens, ' Position RGB': position_timestamp_rgb,
                               ' Position Depth': position_timestamp_depth, ' Labels': sum_FC_labels_30Hz_aligned}
                              for position_timestamp_xsens, position_timestamp_rgb, position_timestamp_depth,
                              sum_FC_labels_30Hz_aligned
                              in zip(position_timestamp_xsens, position_timestamp_rgb, position_timestamp_depth,
                                     sum_FC_labels_30Hz_aligned)]

                    header = ['Position Xsens', ' Position RGB', ' Position Depth', ' Labels']

                    folder = f'Trial_{trial + 1}'

                    # Path for the folders
                    folder_trial = os.path.join(r'C:\Users\diman\PycharmProjects\dissertation\Trials\Trials_Sync_v2',
                                                f'Participant_{participant + 1}', folder)

                    # Path for the txt file
                    path_txt = os.path.join(os.path.join(r'C:\Users\diman\PycharmProjects\dissertation\Trials\Trials_Sync_v2',
                                                         f'Participant_{participant + 1}', folder), 'm_and_b_values.txt')

                    # Create all trials folders, where the csv, png and txt files will be
                    os.makedirs(folder_trial, exist_ok=True)

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
                    plt.savefig(os.path.join(r'C:\Users\diman\PycharmProjects\dissertation\Trials\Trials_Sync_v2',
                                             f'Participant_{participant + 1}', folder) + r'\Sync_Plots.png')

                    # Close all plots, once the trial processing is completed, to decrease the used memory when executing
                    plt.close('all')

            print(f"Total number of '1': {total_num_ones}")
            print(f"Total number of '-1': {total_num_n_ones}")
            print(f"Total number of '0': {total_num_zeros}")

    case '2':
        """----------- VIDEO TO FRAMES EXTRACTION -----------"""
        part = int(input("Enter participant number: "))
        trial = int(input("Enter trial number: "))

        part_trial = f'Participant_{part}\\Trial_{trial}\\gait_rgb_T{trial}_P{part}.avi'
        video_path = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\GAIT_FRAMES'
        output_folder = os.path.join(video_path, f'Participant_{part}\\Trial_{trial}')
        func.extract_images(os.path.join(video_path, part_trial), output_folder)

    case '3':
        """----------- FRAMES LABELING TYPING INTO IMAGE -----------"""
        part = int(input("Enter number of participants to be processed: "))
        trials = int(input("Enter number of trials to be processed: "))

        """----------- ITERATE THROUGH EVERY PARTICIPANT WITH A PROGRESS BAR -----------"""
        for participant in tqdm(range(2, part), desc="Processing Participants"):
            path_zip = os.path.join(directory_participants, contents[participant])

            """----------- OPEN ZIP FILE -----------"""
            with (zipfile.ZipFile(path_zip, 'r') as zip_ref):
                zip_contents = zip_ref.namelist()

                for trial in tqdm(range(2, trials), desc=f"Participant [{participant + 1}] - Processing Trials"):

                    # Get video files
                    video_files = [item for item in zip_contents if item.endswith('avi') and item.__contains__('gait')]
                    sorted_video_files = sorted(video_files, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))
                    corrected_video_files = [item.replace('/', '\\') for item in sorted_video_files]

                    # Sync Files Variables
                    trial_positions = []
                    list_trial_positions = []

                    path_sync_excel = (f'C:\\Users\\diman\\PycharmProjects\\dissertation\\DATA_PROCESSING\\Trials'
                                       f'\\Trials_Sync_v2\\Participant_{participant + 1}')

                    # Get the sync files
                    sync_file = [os.path.join(root, name) for root, dirs, files in os.walk(path_sync_excel)
                                 for name in files if name.endswith(".csv")]
                    sorted_sync_file = sorted(sync_file, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))

                    csv_file = pd.read_csv(sorted_sync_file[trial])
                    positions = csv_file.values

                    # Convert it to a list
                    trial_positions.append(positions.tolist())

                    # Loop that converts a list[list[list[]]] to a list[]
                    for element in trial_positions:
                        for value in element:
                            for i in value:
                                list_trial_positions.append(i)

                    # Get extracted frames
                    dir_frames = os.listdir(
                        f'C:\\Users\\diman\\OneDrive\\Ambiente de Trabalho\\DATASET\\GAIT_FRAMES'
                        f'\\Participant_{participant + 1}\\Trial_{trial + 1}')

                    # Get labels from sync file
                    labels = list_trial_positions[3::4]

                    # Get RGB frames from sync file
                    rgb_frames = list_trial_positions[1::4]

                    i = 0

                    # Loop through the extracted RGB frames
                    for frame in rgb_frames:
                        input_image_path = os.path.join(f'C:\\Users\\diman\\OneDrive\\Ambiente de Trabalho\\DATASET\\'
                                                        f'GAIT_FRAMES\\Participant_{participant + 1}\\Trial_{trial + 1}'
                                                        , dir_frames[frame])
                        output_image_path = (f"C:\\Users\\diman\\OneDrive\\Ambiente de Trabalho\\DATASET\\EVENTS_VIDEOS"
                                             f"\\Participant_{participant + 1}\\Trial_{trial + 1}\\frame_{frame}_labeled.jpg")

                        # Text to be added
                        text = str(labels[i])

                        i += 1

                        # Position where the text will be placed (x, y)
                        position = (30, 30)

                        # Font size
                        font_size = 70

                        # Font color (R, G, B)
                        font_color = (255, 0, 0)

                        func.write_text_on_image(input_image_path, output_image_path, text, position, font_size,
                                                 font_color)
