import pandas as pd
import os
import random
import re
from tqdm import tqdm
import cv2


def CreateBalancedDataset():
    number_participants = 15
    number_trials = 24
    total_labels_df = pd.DataFrame(columns=['Video Path', 'Frames Indexes', 'Class'])

    num_frames_one = 0
    num_frames_n_one = 0
    num_frames_zero = 0

    for participant in tqdm(range(number_participants), desc="Processing Participants"):
        for trial in tqdm(range(number_trials), desc=f"Participant [{participant + 1}] - Processing Trials"):
            trial_dfs = []
            path_labeling = os.path.join(r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\ALIGNED_LABELING',
                                         f'Participant_{participant+1}\Trial_{trial+1}')
            csv_file = pd.read_excel(os.path.join(path_labeling, 'labeling_aligned.xlsx'))
            values = csv_file.values.tolist()

            rgb_frames = []
            labels = []
            total_labels = {'1': 0, '-1': 0, '0': 0}

            start_index = 0
            stop_index = 0

            for row in values:
                rgb_frames.append(row[1])
                labels.append((row[2]))

            rgb_labels = list(zip(rgb_frames, labels))      # tuples containing frame index and label

            for item in rgb_labels:
                if item[1] == 1 or item[1] == -1:
                    start_index = rgb_labels.index(item)
                    break

            for item in rgb_labels[::-1]:
                if item[1] == 1 or item[1] == -1:
                    stop_index = rgb_labels.index(item)
                    break

            start_phase = rgb_labels[:start_index]
            stop_phase = rgb_labels[stop_index+1:]
            gait_phase = rgb_labels[start_index:stop_index]

            # Lists of [1, -1, 0] from gait phase
            list_ones = [item for item in gait_phase if item[1] == 1]
            list_n_ones = [item for item in gait_phase if item[1] == -1]
            list_gait_zeros = [item for item in gait_phase if item[1] == 0]

            min_labels = min(len(list_ones), len(list_n_ones), len(list_gait_zeros))  # for each label [1, -1, 0], per trial
            num_labels_gait = round(min_labels * 0.7)                                 # 70% of labels are from the gait phase
            num_labels_start = num_labels_stop = round((min_labels * 0.3) / 2)        # 30% of labels are from stop and start phases

            # Lists of 0 from gait, start and stop phases
            start_label_zero = random.sample(start_phase, num_labels_start)
            stop_label_zero = random.sample(stop_phase, num_labels_stop)
            gait_label_zero = random.sample(list_gait_zeros, num_labels_gait)

            # Total random labels [1, -1, 0]
            total_label_zero = gait_label_zero + start_label_zero + stop_label_zero
            label_one = random.sample(list_ones, min_labels)
            label_n_one = random.sample(list_n_ones, min_labels)

            # Number of frames
            num_frames_one += len(label_one)
            num_frames_n_one += len(label_n_one)
            num_frames_zero += len(total_label_zero)

            if not len(label_one) == len(label_n_one) == len(total_label_zero):
                print(f'Check trial {trial+1}.\n1: {len(label_one)}\n-1: {len(label_n_one)}\n0: {len(total_label_zero)}')

            total_labels['1'] = [item[0] for item in label_one]
            total_labels['-1'] = [item[0] for item in label_n_one]
            total_labels['0'] = [item[0] for item in total_label_zero]

            path_videos = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\GAIT_VIDEOS'
            contents = os.listdir(path_videos)
            videos = []

            # Iterate over folders and sub-folders to find .avi files
            for folder_path, _, files in os.walk(os.path.join(path_videos, contents[participant])):
                # Check if any .avi files exist in the current folder
                for file in files:
                    if file.endswith('.avi'):
                        videos.append(os.path.join(folder_path, file))
                    else:
                        raise Exception(f'{os.path.join(folder_path, file)} not found.')

            sorted_videos = sorted(videos, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))
            sorted_videos = [item.replace(path_videos+'\\', '') for item in sorted_videos]
            labels_ = [1, -1, 0]

            for label in labels_:
                # Append rows to the DataFrame
                df = pd.DataFrame(
                    {'Video Path': sorted_videos[trial], 'Frames Indexes': [total_labels[str(label)]], 'Class': label})
                trial_dfs.append(df)

            # Concatenate all trial DataFrames for the participant
            participant_labels_df = pd.concat(trial_dfs, ignore_index=True)

            # Append the participant's DataFrame to the total DataFrame
            total_labels_df = pd.concat([total_labels_df, participant_labels_df], ignore_index=True)

    total_frames = num_frames_n_one + num_frames_n_one + num_frames_zero
    # total_labels_df.to_excel(r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\balanced_dataset.xlsx', index=False)

    # Get number of frames
    # with open(r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\dataset_frames_count.txt', 'w') as myFile:
    #    myFile.write(f"Number of frames '1': {num_frames_one}\n")
    #    myFile.write(f"Number of frames '-1': {num_frames_n_one}\n")
    #    myFile.write(f"Number of frames '0': {num_frames_zero}\n")
    #    myFile.write(f"Total number of frames: {total_frames}\n")

    return total_labels_df


class Dataset:
    def __init__(self):
        self.dataset_path = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET'     # alterar para o dataset final (depois de balanceado)
        self.excel_name = 'RGB_labeling_30Hz_balanced_aligned.xlsx'                     # alterar para o dataset final (depois de balanceado)

        self.treino = []
        self.val = []
        self.teste = []

    def SplitDataset(self, num_participants, test_participants, val_participants):
        # df = create_balanced_dataset()
        dataset = pd.read_excel(os.path.join(self.dataset_path, self.excel_name))
        my_data = dataset.values.tolist()                                               # probably not needed
        path_labeling = r'C:\Users\diman\PycharmProjects\dissertation\Labeling'

        print(f'Train size: {(num_participants - len(test_participants) - len(val_participants))/num_participants}; '
              f'Val size: {len(val_participants) / num_participants}; '
              f'Test size: {len(test_participants) / num_participants}')

        test_val_indexes = {'teste': [], 'val': []}
        train_indexes = set(range(dataset.shape[0]))

        for dataset_ids, name in zip([test_participants, val_participants], ['teste', 'val']):
            for subj_id in sorted(dataset_ids):
                subj_path_name = 'participant' + subj_id
                indexes = [ind for ind in range(dataset.shape[0]) if subj_path_name in dataset.iloc[ind, 0]]
                test_val_indexes[name].extend(indexes)
                train_indexes = train_indexes - set(indexes)

        train_indexes = sorted(train_indexes)
        self.treino = dataset.take(train_indexes)
        self.teste = dataset.take(test_val_indexes["teste"])
        self.val = dataset.take(test_val_indexes["val"])

        print('DF_test: ', self.teste)
        print('DF_val: ', self.val)

        print(f'Total of rows in balanced dataset: {dataset.shape[0]}\n'
              f'Number of rows in TRAIN: {self.treino.shape[0]}\n'
              f'Number of rows in VAL: {self.val.shape[0]}\n'
              f'Number of rows in TEST: {self.teste.shape[0]}')

        all_indexes = {'treino': train_indexes, 'val': test_val_indexes['val'], 'teste': test_val_indexes['teste']}

        for item in all_indexes.items():
            count = 0
            for index in item[1]:
                folder = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\GAIT_VIDEOS'
                path = dataset.iloc[index]['Video Path']
                cap = cv2.VideoCapture(os.path.join(folder, path))

                if not cap.isOpened():
                    print("Error: Unable to open video.")

                index_list = dataset.iloc[index]['Frames Indexes'].split()

                for i in index_list:
                    frame_index = int(re.sub('\D', '', i))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()

                    count += 1
                    if dataset.iloc[index]['Class'] == 1:
                        frame_dir = os.path.join(path_labeling, item[0], f'ones\{count}.png')
                        cv2.imwrite(frame_dir, frame)
                        print(f'Image Saved: {frame_index}.png ; ones folder ; {item[0]}')
                    elif dataset.iloc[index]['Class'] == -1:
                        frame_dir = os.path.join(path_labeling, item[0], f'minus_ones\{count}.png')
                        cv2.imwrite(frame_dir, frame)
                        print(f'Image Saved: {frame_index}.png ; minus_ones folder ; {item[0]}')
                    elif dataset.iloc[index]['Class'] == 0:
                        frame_dir = os.path.join(path_labeling, item[0], f'zeros\{count}.png')
                        cv2.imwrite(frame_dir, frame)
                        print(f'Image Saved: {frame_index}.png ; zeros folder ; {item[0]}')

                    print(f'{count}/135299 saved images.')
                    print(f'{round((count / 135299) * 100)} % completed.')
                # Release the video capture object and close windows
                cap.release()
                cv2.destroyAllWindows()

        return self.treino, self.val, self.teste

    # def create_dataset(self):

