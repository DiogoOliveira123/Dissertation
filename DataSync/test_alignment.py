import os
import re
import pandas as pd
import cv2
import sys

NUMBER_PARTICIPANTS = 15
NUMBER_TRIALS = 24

participant = 1
trial = 3

for participant in range(participant - 1, NUMBER_PARTICIPANTS):
    for trial in range(trial - 1, NUMBER_TRIALS):
        directory = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\ALIGNED_LABELING'
        csv_file = pd.read_excel(os.path.join(directory, f'Participant_{participant+1}\Trial_{trial+1}\labeling_aligned.xlsx'))

        values = csv_file.values.tolist()

        rgb_frames = []
        labels = []

        # Gait events
        R_HS_event = []
        R_TO_event = []
        L_HS_event = []
        L_TO_event = []

        # Gait events shifted
        R_HS_event_shifted = []
        R_TO_event_shifted = []
        L_HS_event_shifted = []
        L_TO_event_shifted = []

        for row in values:
            rgb_frames.append(row[1])
            labels.append(row[2])

        for i in range(len(labels) - 1):
            if labels[i] == -1 and labels[i + 1] == 0:
                R_HS_event.append(i + 1)
            elif labels[i] == 0 and labels[i + 1] == -1:
                R_TO_event.append(i + 1)
            elif labels[i] == 1 and labels[i + 1] == -0:
                L_HS_event.append(i + 1)
            elif labels[i] == 0 and labels[i + 1] == 1:
                L_TO_event.append(i + 1)
            elif labels[i] == 1 and labels[i + 1] == -1 or labels[i] == -1 and labels[i + 1] == 1:
                raise Exception('Labeling incorrect! Check if there is a sequence of the wrong label.')

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
        time = 1

        steps = 0
        steps_list = []
        for i in range(rgb_frames[250]):
            steps_list.append(0)

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
            if frame_index + 1 < len(rgb_frames):
                cv2.putText(frames_list[rgb_frames[frame_index]],
                            str(labels[frame_index]),
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),
                            2,
                            cv2.LINE_4)

                # Display the processed frame
                cv2.imshow(f'Trial {trial + 1} - Participant {participant + 1}', frames_list[rgb_frames[frame_index]])
                cv2.waitKey(time)
                print(f'Frame number: {rgb_frames[frame_index]} | Label: {labels[frame_index]}')

                frame_index += 1

            else:
                break

            if rgb_frames[frame_index] > 280:
                time = 25

            # go to wherever frame in the video
            if cv2.waitKey(time) & 0xFF == ord('t'):
                txt = int(input(f"'T' pressed: What frame do you want to go to? range: {rgb_frames[0]} - {len(labels)}\n"))
                if txt < rgb_frames[0] or txt > len(rgb_frames) + 1:
                    raise Exception('Frame out of limit!')
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, rgb_frames[txt])
                    frame_index = txt

            # exit video
            if cv2.waitKey(time) & 0xFF == ord('q'):
                print("'Q' pressed: Trial terminated!")
                break

            # pause video
            if cv2.waitKey(time) & 0xFF == ord('p'):
                print('Video paused.')
                cv2.waitKey(-1)

            # restart program
            if cv2.waitKey(time) & 0xFF == ord('r'):
                os.system('cls')
                os.execv(sys.executable, ['python'] + sys.argv)

        # Release the video capture object and close windows
        cap.release()
        cv2.destroyAllWindows()





