import os
import re
import pandas as pd
import cv2

NUMBER_PARTICIPANTS = 15
NUMBER_TRIALS = 24

participant = 15
trial = 10

path_xlsx = r'C:\Users\diman\PycharmProjects\dissertation\Trials\Trials_Sync_v2'

for participant in range(participant - 1, NUMBER_PARTICIPANTS):
    for trial in range(trial - 1, NUMBER_TRIALS):
        directory = os.path.join(path_xlsx, f'Participant_{participant + 1}', f'Trial_{trial+1}')
        csv_file = pd.read_csv(os.path.join(directory, f'Trial_{trial + 1}.csv'))

        values = csv_file.values.tolist()
        
        # RGB indexes and unaligned labeling
        rgb_frames = []
        labels = []

        # Video frames
        frames_list = []

        # List events
        events = []

        # Gait events shifted
        index_label = []
        
        # Aligned labeling
        labels_aligned = []

        for row in values:
            rgb_frames.append(row[1])
            labels.append((row[3]))

        for i in range(len(labels) - 1):
            if labels[i] == -1 and labels[i + 1] == 0:
                events.append(rgb_frames[i+1])
            elif labels[i] == 0 and labels[i + 1] == -1:
                events.append(rgb_frames[i+1])
            elif labels[i] == 1 and labels[i + 1] == -0:
                events.append(rgb_frames[i+1])
            elif labels[i] == 0 and labels[i + 1] == 1:
                events.append(rgb_frames[i+1])

        path_videos = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\GAIT_VIDEOS'
        contents = os.listdir(path_videos)
        videos = []

        # Iterate over folders and subfolders to find .avi files
        for folder_path, _, files in os.walk(os.path.join(path_videos, contents[participant])):
            # Check if any .avi files exist in the current folder
            for file in files:
                if file.endswith('.avi'):
                    videos.append(os.path.join(folder_path, file))

        sorted_videos = sorted(videos, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))

        # Open the video file
        cap = cv2.VideoCapture(sorted_videos[trial])
        counter = 0

        if not cap.isOpened():
            print("Error: Unable to open video.")

        frame_index = 0
        time = 5

        # Create a window to display the frames
        cv2.namedWindow(f'Trial {trial + 1} - Participant {participant + 1}')

        # List to store all the frames
        frames_list = []

        # Loop through the video frames to fill the frames_list
        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # Check if frame is successfully read
            if not ret:
                print("End of video.")
                break

            # Append the frame to the frames_list
            frames_list.append(frame)

        # Loop through the frames in frames_list
        for frame in frames_list:
            text = ''
            if frame_index + 1 < len(rgb_frames):
                edit = True
                if rgb_frames[frame_index] in events:
                    while edit:
                        txt = input('Move backwards (a), forwards (d), resume video (r) or save event (s)?\n')

                        # move backwards
                        if txt == 'a' or txt == 'A':
                            frame_index -= 1
                            if frame_index < 0:
                                frame_index = 0

                            # Display the frame
                            cv2.imshow(f'Trial {trial + 1} - Participant {participant + 1}', frames_list[rgb_frames[frame_index]])
                            cv2.waitKey(time)
                            print(f'Frame number: {rgb_frames[frame_index]} | Label: {labels[frame_index]}')

                        # move forwards
                        elif txt == 'd' or txt == 'D':
                            frame_index += 1
                            if frame_index >= len(frames_list):
                                frame_index = len(frames_list) - 1

                            # Display the frame
                            cv2.imshow(f'Trial {trial + 1} - Participant {participant + 1}', frames_list[rgb_frames[frame_index]])
                            cv2.waitKey(time)
                            print(f'Frame number: {rgb_frames[frame_index]} | Label: {labels[frame_index]}')

                        # save event
                        elif txt == 's' or txt == 'S':
                            label_in = int(input('Enter label:\n'))
                            index_label.append((frame_index, label_in))
                            print('Event successfully saved!')

                        # resume video
                        elif txt == 'r' or txt == 'R':
                            print('Resume video...')
                            edit = False

                # Display the frame
                cv2.imshow(f'Trial {trial + 1} - Participant {participant + 1}', frames_list[rgb_frames[frame_index]])
                cv2.waitKey(time)
                print(f'Frame number: {rgb_frames[frame_index]} | Label: {labels[frame_index]}')
                frame_index += 1

            else:
                break  # Break out of the loop after one iteration

            # skip trial
            if cv2.waitKey(time) & 0xFF == ord('q'):
                print("Trial terminated!")
                break

        # Release the video capture object and close windows
        cap.release()
        cv2.destroyAllWindows()

        # Reconstruct labeling with shifted events
        for i in range(len(labels)):
            labels_aligned.append(0)

        for i in range(len(index_label)):
            if i + 1 < len(index_label):
                curr_index, curr_label = index_label[i]
                next_index, next_label = index_label[i+1]
                diff = next_index - curr_index

                for item in range(diff):
                    labels_aligned[curr_index+item] = curr_label

        # Write indexes of the frames correspondent to the aligned events
        path_txt = os.path.join(r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\ALIGNED_LABELING',
                                f'Participant_{participant+1}\Trial_{trial+1}\indexes_shifted_events.txt')

        with open(path_txt, 'w') as file:
            file.write(str(index_label))

        # Write aligned labeling to xlsx file
        path = os.path.join(r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\ALIGNED_LABELING',
                                f'Participant_{participant+1}\Trial_{trial+1}\labeling_aligned.xlsx')

        df = pd.DataFrame({'RGB': rgb_frames, 'Labels': labels_aligned})

        df.to_excel(path)

