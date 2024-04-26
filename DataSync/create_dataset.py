import pandas as pd
import os
import random
import re
from tqdm import tqdm
import cv2
import numpy as np


def CreateBalancedDataset():
    number_participants = 15
    number_trials = 24
    total_labels_df = pd.DataFrame(columns=['Video Path', 'Frames Indexes', 'Class'])

    num_frames_one = 0
    num_frames_n_one = 0
    num_frames_zero = 0

    total_gait_frames = 0

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

            # Number of frames in gait phase
            total_gait_frames += len(gait_phase)

            # Lists of [1, -1, 0] from gait phase
            list_ones = [item for item in gait_phase if item[1] == 1]
            list_n_ones = [item for item in gait_phase if item[1] == -1]
            list_gait_zeros = [item for item in gait_phase if item[1] == 0]

            num_frames_one += len(list_ones)
            num_frames_n_one += len(list_n_ones)
            num_frames_zero += len(list_gait_zeros)

            min_labels = 60                                                           # for each label [1, -1, 0], per trial
            num_labels_gait = round(min_labels * 0.5)                                 # 70% of labels are from the gait phase
            num_labels_start = num_labels_stop = round((min_labels * 0.5) / 2)        # 30% of labels are from stop and start phases

            # Lists of 0 from gait, start and stop phases
            start_label_zero = random.sample(start_phase, num_labels_start)
            stop_label_zero = random.sample(stop_phase, num_labels_stop)
            gait_label_zero = random.sample(list_gait_zeros, num_labels_gait)

            # Total random labels [1, -1, 0]
            total_label_zero = gait_label_zero + start_label_zero + stop_label_zero
            label_one = random.sample(list_ones, min_labels)
            label_n_one = random.sample(list_n_ones, min_labels)

            # Number of frames
            # num_frames_one += len(label_one)
            # num_frames_n_one += len(label_n_one)
            # num_frames_zero += len(total_label_zero)

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
        # myFile.write(f"Number of frames '1': {num_frames_one}\n")
        # myFile.write(f"Number of frames '-1': {num_frames_n_one}\n")
        # myFile.write(f"Number of frames '0': {num_frames_zero}\n")
        # myFile.write(f"Total number of frames: {total_frames}\n")

    # return total_labels_df, total_frame
    return total_gait_frames, num_frames_one, num_frames_n_one, num_frames_zero


# Crop to Region of Interest (ROI):
def crop_ROI(img, new_img_shape=(224, 224), height_leftcorner=60, height_rightcorner=450, width_leftcorner=85, width_rightcorner=365):

    new_img = np.zeros((new_img_shape[1], new_img_shape[0], img.shape[-1]))  # numpy: (height, width, channels)
    img = img[height_leftcorner:height_rightcorner, width_leftcorner:width_rightcorner]

    if img.shape[1] > img.shape[0]:
        # always choose the biggest axis of the cropped image and equal that to the correspondent new_image axis length
        # even if that new_axis' length is the smallest of the new image shape that we want
        # because, like this, the crop img will be resized in a way that its other (smaller) axis will be smaller than its new biggest axis' length and, therefore,
        # smaller than the new_image axis' lengths (both of them), avoiding a new image that, despite maintaining the aspect ratio, would surpass the stipulated
        # shape for the img (new_img.shape)
        # than we can just fill the extra pixels with 0
        # width is bigger than height

        fixed_width = new_img_shape[1]
        percent = (fixed_width / float(img.shape[1]))
        height = int((float(img.shape[0]) * float(percent)))
        img = cv2.resize(img, dsize=(fixed_width, height), interpolation=cv2.INTER_AREA)  # (width, height)
        if img.shape != new_img.shape:
            border = int((new_img.shape[0] - height)/2)         # ATTENTION: IT SHOULD BE PAIR
            new_img[border:-border, :img.shape[1]] = img
        else:
            new_img = img
    else:
        fixed_height = new_img_shape[0]
        percent = (fixed_height / float(img.shape[0]))
        width = int((float(img.shape[1]) * float(percent)))
        img = cv2.resize(img, dsize=(width, fixed_height), interpolation=cv2.INTER_AREA)  # (width, height)
        if len(img.shape) < len(new_img.shape):     # if no_channels=1, cv2_resize removes that axis
            img = img[:, :, np.newaxis]
        if img.shape != new_img.shape:
            border = int((new_img.shape[1] - width)/2)  # ATTENTION: IT SHOULD BE PAIR
            new_img[:img.shape[0], border:-border] = img
        else:
            new_img = img

    return new_img


def cut_original_img(img, new_img_shape=(224, 224)):
    # to maintain the aspect ratio of the original img, when resizing
    # applied when original aspect ratio > the new one, so we can cut the img

    img = np.array(img, dtype=np.float32)

    if (img.shape[1]/img.shape[0]) > (new_img_shape[1]/new_img_shape[0]):   # (width/height), assuming width >= height
        new_ratio = int(new_img_shape[1]/new_img_shape[0])
        width = new_ratio * img.shape[0]
        border = int((img.shape[1] - width)/2)
        img = img[:, border:-border]                                        # cuts BG, so it doesn't matter

    return img


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
        path_labeling = r'/Labeling'

        crop = True

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
                    # resized_frame = crop_ROI(frame)
                    resized_frame = cut_original_img(frame, new_img_shape=(224, 224))
                    if crop:
                        resized_frame = crop_ROI(resized_frame, new_img_shape=(224, 224))
                    else:
                        resized_frame = cv2.resize(resized_frame, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                        if len(resized_frame.shape) < len(frame.shape):
                            resized_frame = resized_frame[:, :, np.newaxis]

                    count += 1
                    if dataset.iloc[index]['Class'] == 1:
                        frame_dir = os.path.join(path_labeling, item[0], f'ones\{count}.png')
                        cv2.imwrite(frame_dir, resized_frame)
                        print(f'Image Saved: {frame_index}.png ; ones folder ; {item[0]}')
                    elif dataset.iloc[index]['Class'] == -1:
                        frame_dir = os.path.join(path_labeling, item[0], f'minus_ones\{count}.png')
                        cv2.imwrite(frame_dir, resized_frame)
                        print(f'Image Saved: {frame_index}.png ; minus_ones folder ; {item[0]}')
                    elif dataset.iloc[index]['Class'] == 0:
                        frame_dir = os.path.join(path_labeling, item[0], f'zeros\{count}.png')
                        cv2.imwrite(frame_dir, resized_frame)
                        print(f'Image Saved: {frame_index}.png ; zeros folder ; {item[0]}')

                    print(f'Index Number: {index}')
                    print(f'{count}/64800 saved images.')
                    print(f'{round((count / 64800) * 100)} % completed.')
                # Release the video capture object and close windows
                cap.release()

        return self.treino, self.val, self.teste

    # def create_dataset(self):

