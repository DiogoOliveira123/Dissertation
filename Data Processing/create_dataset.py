import pandas as pd
import os
from tqdm import tqdm

class Dataset:

    def __init__(self):
        self.dataset_path = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET'
        self.excel_name = 'RGB_labeling_30Hz_aligned_indexes.xlsx'

        self.train = []
        self.val = []
        self.test = []

    def balance_dataset(self):
        csv_file = pd.read_excel(os.path.join(self.dataset_path, self.excel_name))
        values = csv_file.values.tolist()

        rgb_frames = []
        labels = []

        for row in values:
            rgb_frames.append(row[1])
            labels.append((row[2]))

        # Get number of '1', '-1' and '0'
        num_ones = labels.count(1)
        num_n_ones = labels.count(-1)
        num_zeros = labels.count(0)

        print('Class 0: ', num_zeros)
        print('Class 1: ', num_ones)
        print('Class -1: ', num_n_ones)

        min_label = min(num_ones, num_n_ones, num_zeros)



    def split_dataset(self):

    def create_dataset(self):


