import os
import pandas as pd
from itertools import groupby
import matplotlib.pyplot as plt

NUMBER_PARTICIPANTS = 15
NUMBER_TRIALS = 24

for participant in range(1, NUMBER_PARTICIPANTS):
    # Create subplots for each trial in a 4x3 grid
    fig1, axs1 = plt.subplots(4, 3, figsize=(15, 20))  # Adjust figsize for larger subplots
    fig2, axs2 = plt.subplots(4, 3, figsize=(15, 20))  # Adjust figsize for larger subplots

    # File with participant tests info
    log_info = []

    for trial in range(NUMBER_TRIALS):
        # Calculate row and column indices for the subplot
        if trial < 12:
            row_idx = trial // 3  # Integer division to determine the row index
            col_idx = trial % 3  # Modulus operation to determine the column index
        else:
            row_idx = (trial - 12) // 3  # Integer division to determine the row index
            col_idx = (trial - 12) % 3  # Modulus operation to determine the column index

        # Initialize empty lists to store label counts for the current trial
        trial_counts = {'1': 0, '-1': 0, '0': 0}

        path = os.path.join(r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\ALIGNED_LABELING',
                            f'Participant_{participant + 1}\Trial_{trial + 1}')
        directories = os.listdir(path)
        csv_file = pd.read_excel(os.path.join(path, 'labeling_aligned.xlsx'))

        values = csv_file.values.tolist()

        labels = []

        for row in values:
            labels.append(row[2])

        num_ones = labels.count(1)
        num_minus_ones = labels.count(-1)
        num_zeros = labels.count(0)

        trial_counts['1'] = num_ones
        trial_counts['-1'] = num_minus_ones
        trial_counts['0'] = num_zeros

        grouped = [(key, len(list(group))) for key, group in groupby(labels)]

        ones = 0
        minus_ones = 0
        for key in grouped:
            if key[0] == -1:
                minus_ones += 1
            elif key[0] == 1:
                ones += 1

        print(f'Testing Trial {trial + 1}...')
        log_info.append(f'Trial {trial + 1}\n')
        if abs(ones - minus_ones) > 1:
            print('Steps Counting Test --------------->	FAILED')
            log_info.append('Steps Counting Test --------------->	FAILED\n')
        else:
            print('Steps Counting Test --------------->	PASSED')
            log_info.append('Steps Counting Test --------------->	PASSED\n')

        if len(directories) == 2:
            print('Files Counting Test (xlsx and txt) -> 	PASSED')
            log_info.append('Files Counting Test (xlsx and txt) -> 	PASSED\n')
        else:
            print('Files Counting Test (xlsx and txt) -> 	FAILED')
            log_info.append('Files Counting Test (xlsx and txt) -> 	FAILED\n')

        counter = 0
        # Get index of events from labeling_aligned.xlsx
        for i in range(len(labels) - 1):
            if not labels[i] == labels[i+1]:
                counter += 1

        with open(os.path.join(path, 'indexes_shifted_events.txt'), 'r') as txt_file:
            indexes = txt_file.read()
            num_indexes = indexes.count('(')

        if not counter == num_indexes:
            print('Indexes Counting Test -------------> 	FAILED\n')
            log_info.append('Indexes Counting Test -------------> 	FAILED\n')
        else:
            print('Indexes Counting Test -------------> 	PASSED\n')
            log_info.append('Indexes Counting Test -------------> 	PASSED\n')

        log_info.append('\n')

        # Plot histogram for the current trial
        if trial < 12:
            axs1[row_idx, col_idx].bar(trial_counts.keys(), trial_counts.values())
            axs1[row_idx, col_idx].set_title(f'Trial {trial + 1} Label Counts')
            axs1[row_idx, col_idx].set_xlabel('Label')
            axs1[row_idx, col_idx].set_ylabel('Count')

        else:
            axs2[row_idx, col_idx].bar(trial_counts.keys(), trial_counts.values())
            axs2[row_idx, col_idx].set_title(f'Trial {trial + 1} Label Counts')
            axs2[row_idx, col_idx].set_xlabel('Label')
            axs2[row_idx, col_idx].set_ylabel('Count')

    # Save participant labeling histogram
    path_part = os.path.join(r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\ALIGNMENT_TESTS', f'Participant {participant+1}')
    fig1.savefig(os.path.join(path_part, 'labeling_histogram_1_12.png'))
    fig2.savefig(os.path.join(path_part, 'labeling_histogram_13_24.png'))

    plt.close(fig1)
    plt.close(fig2)

    with open(os.path.join(path_part, 'log_file.txt'), 'w') as file:
        for row in log_info:
            file.write(row)
