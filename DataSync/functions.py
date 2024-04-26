
"""------------ FUNCTIONS --------------"""
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont


def get_initial_timestamp(my_str, char1, char2):
    """
    Get timestamp (xxxxxxxxx.xxxxxxxxx) value and replace '_' with a '.'
    :param my_str: string to be processed
    :param char1: character that delimits the desired part of the string (to the right)
    :param char2: character that delimits the desired part of the string (to the left)
    :return: desired part of the string
    """
    underscore_indices = [i for i, x in enumerate(my_str[::-1]) if x == char1]
    dot_stamp_index = my_str.find(char2)
    if len(underscore_indices) >= 2 and dot_stamp_index != -1:
        second_underscore_index = len(my_str) - underscore_indices[1] - 1
        result = my_str[second_underscore_index + 1:dot_stamp_index]
        result_str = ''.join(result)
        timestamp = result_str.replace('_', '.')
        return timestamp
    else:
        print(f"There are not enough '_' or '.' in the string: {my_str}")


def get_depth_timestamps(my_str, char1, char2):
    """
    Get timestamp (xxxxxxxxx.xxxxxxxxx) value and replace '_' with a '.'
    :param my_str: string to be processed
    :param char1: character that delimits the desired part of the string (to the right)
    :param char2: character that delimits the desired part of the string (to the left)
    :return: desired part of the string
    """
    timestamps = []
    for element in my_str:
        underscore_indices = [i for i, x in enumerate(element[::-1]) if x == char1]
        dot_stamp_index = element.find(char2)
        if len(underscore_indices) >= 2 and dot_stamp_index != -1:
            second_underscore_index = len(element) - underscore_indices[1] - 1
            result = element[second_underscore_index + 1:dot_stamp_index]
            result_str = ''.join(result)
            str_replace_point = result_str.replace('_', '.')
            timestamps.append(float(str_replace_point))
        else:
            print(f"There are not enough '_' or '.' in the string: {my_str}")

    return timestamps


def get_rgb_timestamps(zip_ref, my_list):
    """
    Get the timestamps from the RGB data.
    :param zip_ref: zip file to be opened
    :param my_list: RGB list to be processed
    :return: list of RGB timestamps
    """
    rgb_trial_timestamps = []
    with zip_ref.open(my_list) as txt_file:
        lines = txt_file.readlines()
        # Decode bytes to strings
        timestamps_rgb_folders = [line.decode('utf-8') for line in lines]
        for strings in timestamps_rgb_folders:
            # Remove \n from timestamps
            rgb_trial_timestamps.append(float(strings.strip()))
    return rgb_trial_timestamps


def compare_timestamps(my_list1, my_list2, highest_list, lowest_list_length):
    """
    Compare timestamps from different data sources.
    :param my_list1: xsens 30 Hz timestamps list.
    :param my_list2: depth timestamps or rgb timestamps lists, depending on the highest initial timestamp.
    :param highest_list: list with the highest initial timestamp.
    :param lowest_list_length: length of the smallest list.
    :return: lists of aligned timestamps and the position of each aligned timestamp in a determined list
    """
    position_timestamp_list1 = []
    timestamp_list1 = []
    position_timestamp_list2 = []
    timestamp_list2 = []

    for timestamp in range(lowest_list_length):
        closest_value_list1 = min(my_list1, key=lambda x: abs(highest_list[timestamp] - x))
        position_list1 = my_list1.index(closest_value_list1)
        position_timestamp_list1.append(position_list1)
        timestamp_list1.append(closest_value_list1)

        closest_value_list2 = min(my_list2, key=lambda x: abs(highest_list[timestamp] - x))
        position_list2 = my_list2.index(closest_value_list2)
        position_timestamp_list2.append(position_list2)
        timestamp_list2.append(closest_value_list2)

    return (position_timestamp_list1, timestamp_list1,
            position_timestamp_list2, timestamp_list2)


def extract_images(video_path, output_folder):
    """
    Extract frames from a video.
    :param video_path: path to the video to be processed
    :param output_folder: path to where the frames will be saved
    :return: no return
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get the frames per second (fps) and frame width/height
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create the output folder if it doesn't exist
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through frames and save them as images
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as an image
        image_filename = f"{output_folder}/frame_{frame_count:04d}.jpg"
        cv2.imwrite(image_filename, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Frames extracted successfully: {frame_count}")


def write_text_on_image(input_image_path, output_image_path, text, position=(10, 10), font_size=20,
                        font_color=(255, 255, 255), font_path="arial.ttf"):
    """
    Execute the labeling to a set of given frames, by writing the label into the top left corner of the image.
    :param input_image_path: path to the extracted frames
    :param output_image_path: path to the labeled frames
    :param text: text to write into the frame
    :param position: coordinates to write the text
    :param font_size: size of the text
    :param font_color: color of the text
    :param font_path: type of letter of the text
    :return: no return
    """
    # Open the image
    img = Image.open(input_image_path)

    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Set the font size directly by creating a font object
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # Draw text on the image with the specified font size
    draw.text(position, text, font=font, fill=font_color)

    # Save the result
    img.save(output_image_path)


def process_video(path_video, num_participant, num_trial, frame_index):
    contents = os.listdir(path_video)
    videos = []

    # Iterate over folders and subfolders to find .avi files
    for folder_path, _, files in os.walk(os.path.join(path_video, contents[num_participant])):
        # Check if any .avi files exist in the current folder
        for file in files:
            if file.endswith('.avi'):
                videos.append(os.path.join(folder_path, file))

    # Open the video file
    cap = cv2.VideoCapture(videos[num_trial])

    if not cap.isOpened():
        print("Error: Unable to open video.")

    # Loop through the video frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if frame is successfully read
        if not ret:
            print("End of video.")
            break

        return frame

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()





