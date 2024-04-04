import os
import pandas as pd
from tqdm import tqdm

path_xlsx = r'C:\Users\diman\PycharmProjects\dissertation\Trials\Trials_Sync_v2'

NUMBER_PARTICIPANTS = 15
NUMBER_TRIALS = 24

total_num_ones = 0
total_num_n_ones = 0
total_num_zeros = 0

total_rgb_frames = []
total_labels = []

participants_index = []
count = 0

for participant in range(NUMBER_PARTICIPANTS):
    count += 1
    for trial in tqdm(range(NUMBER_TRIALS), desc=f"Participant [{participant + 1}] - Processing Trials"):
        directory = os.path.join(path_xlsx, f'Participant_{participant + 1}', f'Trial_{trial+1}')

        csv_file = pd.read_csv(os.path.join(directory, f'Trial_{trial+1}.csv'))
        values = csv_file.values.tolist()

        rgb_frames = []
        labels = []

        for row in values:
            rgb_frames.append(row[1])
            labels.append((row[3]))

        total_rgb_frames.extend(rgb_frames)
        total_labels.extend(labels)

        for i in range(len(labels)):
            participants_index.append(count)

df = pd.DataFrame({'Participant ': participants_index, 'RGB': total_rgb_frames, 'Labels': total_labels})

df = df.set_index('Participant ')

df.to_excel(r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\dataset.xlsx')












