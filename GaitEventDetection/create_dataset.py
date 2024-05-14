import pandas as pd
import os
import re
from tqdm import tqdm
import cv2
import numpy as np


def getFolders(directory, num):
    """
    Used in Dataset.CreateDataset() to split the train, val and test paths.
    :param directory: base path to train, val and test directories.
    :param num: number referring to train, val or test (0, 1, 2)
    :return: path of train, val or test
    """
    contents = os.listdir(directory)
    path = []

    for folder_path, _, _ in os.walk(os.path.join(directory, contents[num])):
        path.append(folder_path)

    path = [item for item in path if 'Trial' in item]
    path = sorted(path, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))
    path = sorted(path, key=lambda x: int(re.search(r'Participant_(\d+)', x).group(1)))

    return path


class Dataset:
    def __init__(self):
        self.dataset_path = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET'
        self.excel_name = 'RGB_labeling_30Hz_balanced_aligned_v4.xlsx'

        self.train = []
        self.val = []
        self.test = []

    @staticmethod
    def CreateDataset():
        """
        Creates a CSV file, with the entire dataset, divided into sequences.
        :return: Dataframe containing the entire dataset.
        """
        number_participants = 15
        number_trials = 24
        total_labels_df = pd.DataFrame(columns=['Frames Path', 'Frames Indexes', 'Class'])

        num_frames_one = 0
        num_frames_n_one = 0
        num_frames_zero = 0

        num_total_labels = 0

        train_num = 0
        val_num = 1
        test_num = 2

        directory = r'D:\Labeling_v2'

        train_path = getFolders(directory, train_num)
        val_path = getFolders(directory, val_num)
        test_path = getFolders(directory, test_num)

        for participant in tqdm(range(number_participants), desc="Processing Participants"):
            for trial in tqdm(range(number_trials), desc=f"Participant [{participant + 1}] - Processing Trials"):
                path_labeling = os.path.join(r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\ALIGNED_LABELING',
                                             f'Participant_{participant + 1}\Trial_{trial + 1}')
                csv_file = pd.read_excel(os.path.join(path_labeling, 'labeling_aligned.xlsx'))
                values = csv_file.values.tolist()

                rgb_frames = []
                labels = []

                start_index = 0
                stop_index = 0

                for row in values:
                    rgb_frames.append(row[1])
                    labels.append((row[2]))

                rgb_labels = list(zip(rgb_frames, labels))  # tuples containing frame index and label

                for item in rgb_labels:
                    if item[1] == 1 or item[1] == -1:
                        start_index = rgb_labels.index(item)
                        break

                for item in rgb_labels[::-1]:
                    if item[1] == 1 or item[1] == -1:
                        stop_index = rgb_labels.index(item)
                        break

                start_phase = rgb_labels[:start_index]
                stop_phase = rgb_labels[stop_index:]
                gait_phase = rgb_labels[start_index:stop_index]

                num_frames_start = num_frames_stop = 30

                total_trial = start_phase[(len(start_phase) - 1) - num_frames_start:] + gait_phase + stop_phase[
                                                                                                     :num_frames_stop]

                for item in total_trial:
                    if item[1] == 1:
                        num_frames_one += 1
                    elif item[1] == -1:
                        num_frames_n_one += 1
                    else:
                        num_frames_zero += 1

                num_total_labels += len(total_trial)

                # Sliding Window Method
                window_length = 4

                for i in range(window_length - 1, len(total_trial)):
                    indexes_df = [[total_trial[i - 3][0], total_trial[i - 2][0], total_trial[i - 1][0], total_trial[i][0]]]
                    next_label = total_trial[i][1]

                    part_trial = f'Participant_{participant + 1}\\Trial_{trial + 1}'

                    current_path = ''

                    for path in train_path:
                        if part_trial in path:
                            current_path = path
                            break  # Exit the loop once the path is found

                    for path in val_path:
                        if part_trial in path:
                            current_path = path
                            break  # Exit the loop once the path is found

                    for path in test_path:
                        if part_trial in path:
                            current_path = path
                            break  # Exit the loop once the path is found

                    # Create DataFrame
                    df = pd.DataFrame(
                        {'Frames Path': current_path,
                         'Frames Indexes': indexes_df,
                         'Class': next_label})

                    # Concatenate all trial DataFrames for the participant
                    total_labels_df = total_labels_df.append(df, ignore_index=True)

        total_labels_df.to_excel(
            r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\RGB_labeling_30Hz_balanced_aligned_v4.xlsx',
            index=False)

        # Get number of frames
        with open(r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\dataset_frames_count_v5.txt', 'w') as myFile:
            myFile.write(f"Number of frames '1': {num_frames_one}\n")
            myFile.write(f"Number of frames '-1': {num_frames_n_one}\n")
            myFile.write(f"Number of frames '0': {num_frames_zero}\n")
            myFile.write(f"Total number of frames: {num_total_labels}\n")

        return total_labels_df

    # Crop to Region of Interest (ROI):
    @staticmethod
    def crop_ROI(img, new_img_shape=(224, 224), height_leftcorner=60, height_rightcorner=450, width_leftcorner=85,
                 width_rightcorner=365):
        """
        Crops the image into the ROI (Region of Interest)
        :param img: frame
        :param new_img_shape: image shape
        :param height_leftcorner: left corner
        :param height_rightcorner: right corner
        :param width_leftcorner: width of the left corner
        :param width_rightcorner: width of the right corner
        :return: cropped image
        """

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
                border = int((new_img.shape[0] - height) / 2)  # ATTENTION: IT SHOULD BE PAIR
                new_img[border:-border, :img.shape[1]] = img
            else:
                new_img = img
        else:
            fixed_height = new_img_shape[0]
            percent = (fixed_height / float(img.shape[0]))
            width = int((float(img.shape[1]) * float(percent)))
            img = cv2.resize(img, dsize=(width, fixed_height), interpolation=cv2.INTER_AREA)  # (width, height)
            if len(img.shape) < len(new_img.shape):  # if no_channels=1, cv2_resize removes that axis
                img = img[:, :, np.newaxis]
            if img.shape != new_img.shape:
                border = int((new_img.shape[1] - width) / 2)  # ATTENTION: IT SHOULD BE PAIR
                new_img[:img.shape[0], border:-border] = img
            else:
                new_img = img

        return new_img

    @staticmethod
    def cut_original_img(img, new_img_shape=(224, 224)):
        # to maintain the aspect ratio of the original img, when resizing
        # applied when original aspect ratio > the new one, so we can cut the img

        img = np.array(img, dtype=np.float32)

        if (img.shape[1] / img.shape[0]) > (
                new_img_shape[1] / new_img_shape[0]):  # (width/height), assuming width >= height
            new_ratio = int(new_img_shape[1] / new_img_shape[0])
            width = new_ratio * img.shape[0]
            border = int((img.shape[1] - width) / 2)
            img = img[:, border:-border]  # cuts BG, so it doesn't matter

        return img

    def SplitDataset(self, num_participants, test_participants, val_participants):
        """
        Splits and saves the frames of the dataset into the train, val and test folders.
        Each folder contains the respective participants, each with 24 trials.
        :param num_participants: number of total participants
        :param test_participants: number of test participants
        :param val_participants: number of val participants
        :return: indexes of train, val and test rows.
        """
        dataset = pd.read_excel(os.path.join(self.dataset_path, self.excel_name))
        path_labeling = r'D:\Labeling_v3'

        crop = True
        create_dirs = False

        print(f'Train size: {(num_participants - len(test_participants) - len(val_participants)) / num_participants}; '
              f'Val size: {len(val_participants) / num_participants}; '
              f'Test size: {len(test_participants) / num_participants}')

        test_val_indexes = {'test': [], 'val': []}
        train_indexes = set(range(dataset.shape[0]))

        for dataset_ids, name in zip([test_participants, val_participants], ['test', 'val']):
            for subj_id in sorted(dataset_ids):
                subj_path_name = 'Participant_' + subj_id
                indexes = [ind for ind in range(dataset.shape[0]) if subj_path_name in dataset.iloc[ind, 0]]
                test_val_indexes[name].extend(indexes)
                train_indexes = train_indexes - set(indexes)

        train_indexes = sorted(train_indexes)
        self.train = dataset.take(train_indexes)
        self.test = dataset.take(test_val_indexes["test"])
        self.val = dataset.take(test_val_indexes["val"])

        print('DF_test: ', self.test)
        print('DF_val: ', self.val)

        print(f'Total of rows in balanced dataset: {dataset.shape[0]}\n'
              f'Number of rows in TRAIN: {self.train.shape[0]}\n'
              f'Number of rows in VAL: {self.val.shape[0]}\n'
              f'Number of rows in TEST: {self.test.shape[0]}')

        folder_part_df = {
            'train': [part + 1 for part in range(15) if val_participants.count(str(part + 1)) == 0 and
                      test_participants.count(str(part + 1)) == 0],
            'val': list(map(int, val_participants)),
            'test': list(map(int, test_participants))}

        if create_dirs:
            for row in folder_part_df.items():
                for participant in row[1]:
                    os.makedirs(os.path.join(r'D:\Labeling_v3', row[0], f'Participant_{participant}'))
                    for trial in range(24):
                        os.makedirs(os.path.join(r'D:\Labeling_v3',
                                                 row[0],
                                                 f'Participant_{participant}',
                                                 f'Trial_{trial + 1}'))

        all_indexes = {'train': train_indexes, 'val': test_val_indexes['val'], 'test': test_val_indexes['test']}

        # LOOP THROUGH TRAIN, VAL AND TEST
        for item in all_indexes.items():
            indexes = item[1]
            count = 0
            # LOOP THROUGH EACH PARTICIPANT OF TRAIN, VAL AND TEST
            for participants in folder_part_df.items():
                for participant in participants[1]:
                    for trial in range(24):
                        print(f'Processing Trial {trial + 1} - Participant {participant} -> {item[0]} folder')

                        trial_list = []
                        first = True
                        trial_folder = f'Trial_{trial + 1}'
                        # GET THE INDEXES FOR THE GIVEN TRIAL AND PARTICIPANT
                        for index in indexes:
                            if ((f'Participant_{participant}\\' in dataset.iloc[
                                index, 0] or f'Participant_0{participant}\\' in dataset.iloc[
                                    index, 0]) and trial_folder in dataset.iloc[index, 0]
                                    and len(trial_folder) == len(dataset.iloc[index, 0][dataset.iloc[index, 0].find(trial_folder):])):
                                trial_list.append(index)

                        # Iterate over folders and sub-folders to find .avi files
                        path_videos = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\GAIT_VIDEOS'
                        contents = os.listdir(path_videos)
                        videos = []
                        for folder_path, _, files in os.walk(os.path.join(path_videos, contents[participant - 1])):
                            # Check if any .avi files exist in the current folder
                            for file in files:
                                if file.endswith('.avi'):
                                    videos.append(os.path.join(folder_path, file))
                                else:
                                    raise Exception(f'{os.path.join(folder_path, file)} not found.')

                        sorted_videos = sorted(videos, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))
                        sorted_videos = [item.replace(path_videos + '\\', '') for item in sorted_videos]

                        folder = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\GAIT_VIDEOS'
                        path = sorted_videos[trial]
                        cap = cv2.VideoCapture(os.path.join(folder, path))

                        if not cap.isOpened():
                            print("Error: Unable to open video.")

                        for idx in trial_list:
                            index_list = dataset.iloc[idx]['Frames Indexes'].split()
                            # FIRST LOOP PRINTS ALL THE 4 FRAMES OF THE SEQUENCE, BUT DOES IT ONLY ONCE
                            if first:
                                for i in index_list:
                                    frame_index = int(re.sub('\D', '', i))
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                                    ret, frame = cap.read()
                                    resized_frame = self.cut_original_img(frame, new_img_shape=(224, 224))
                                    if crop:
                                        resized_frame = self.crop_ROI(resized_frame, new_img_shape=(224, 224))
                                    else:
                                        resized_frame = cv2.resize(resized_frame, dsize=(224, 224),
                                                                   interpolation=cv2.INTER_AREA)
                                        if len(resized_frame.shape) < len(frame.shape):
                                            resized_frame = resized_frame[:, :, np.newaxis]
                                    count += 1
                                    frame_dir = os.path.join(path_labeling,
                                                             item[0],
                                                             f'Participant_{participant}',
                                                             f'Trial_{trial + 1}',
                                                             f'{frame_index}.jpg')
                                    cv2.imwrite(frame_dir, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                                    print(
                                        f'Image Saved: {frame_index}.png ; {trial + 1}/24 Trials - Participant {participant} ; {item[0]} folder')
                                    print(f'{count}/193252 saved images.')
                                    print(f'{round((count / 193252) * 100)} % completed.')

                                first = False

                            # PRINT THE index_list[1], index_list[2] AND index_list[3], FOR THE REST OF THE FRAMES
                            else:
                                for i in range(1, len(index_list)):
                                    frame_index = int(re.sub('\D', '', index_list[i]))
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                                    ret, frame = cap.read()
                                    resized_frame = self.cut_original_img(frame, new_img_shape=(224, 224))
                                    if crop:
                                        resized_frame = self.crop_ROI(resized_frame, new_img_shape=(224, 224))
                                    else:
                                        resized_frame = cv2.resize(resized_frame, dsize=(224, 224),
                                                                   interpolation=cv2.INTER_AREA)
                                        if len(resized_frame.shape) < len(frame.shape):
                                            resized_frame = resized_frame[:, :, np.newaxis]
                                    count += 1

                                    frame_dir = os.path.join(path_labeling,
                                                             item[0],
                                                             f'Participant_{participant}\Trial_{trial + 1}',
                                                             f'{frame_index}.jpg')
                                    cv2.imwrite(frame_dir, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                                    print(
                                        f'Image Saved: {frame_index}.png ; {trial + 1}/24 Trials - Participant {participant} ; {item[0]} folder')
                                    print(f'{count}/193252 saved images.')
                                    print(f'{round((count / 193252) * 100)} % completed.')

                        # Release the video capture object and close windows
                        cap.release()

        return self.train, self.val, self.test
