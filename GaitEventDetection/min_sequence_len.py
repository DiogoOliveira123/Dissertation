import os
import numpy as np

NUM_PARTICIPANTS = 15
NUM_TRIALS = 24
path = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\ALIGNED_LABELING'

list_dir = []
min_idx = []

for participant in range(NUM_PARTICIPANTS):
    for trial in range(NUM_TRIALS):
        directory = os.listdir(os.path.join(path, f'Participant_{participant+1}', f'Trial_{trial+1}'))
        # list_dir.append(directory)
        file_dir = os.path.join(path, f'Participant_{participant+1}', f'Trial_{trial+1}', directory[0])
        with open(file_dir, 'r') as file:
            file_info = file.read()
            list_of_tuples = eval(file_info)
            list_index = [item[0] for item in list_of_tuples]
            np_list_of_tuples = np.array(list_index)
            diff = np.diff(np_list_of_tuples)
            min_idx.append(diff.min())

print(f'Min sequence length: {min(min_idx)}')
print(f'Number of occurrences: {min_idx.count(3)}')