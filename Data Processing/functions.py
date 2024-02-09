

"""------------ FUNCTIONS --------------"""


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
    :param my_list1: Xsens 30 Hz timestamps list.
    :param my_list2: Depth timestamps or rgb timestamps lists, depending on the highest initial timestamp.
    :param highest_list: List with the highest initial timestamp.
    :param lowest_list_length: Value of the lowest length of the lists to be compared.
    :return: lists of aligned timestamps and the position of each aligned timestamp in a determined list
    """
    position_timestamp_list1 = []
    timestamp_list1 = []
    position_timestamp_list2 = []
    timestamp_list2 = []
    result_position_timestamp_list1 = []
    result_position_timestamp_list2 = []
    result_timestamp_list1 = []
    result_timestamp_list2 = []

    for timestamp in range(lowest_list_length):
        closest_value_list1 = min(my_list1, key=lambda x: abs(highest_list[timestamp] - x))
        position_list1 = my_list1.index(closest_value_list1)
        position_timestamp_list1.append(position_list1)
        timestamp_list1.append(closest_value_list1)

    for timestamp in range(lowest_list_length):
        closest_value_list2 = min(my_list2, key=lambda x: abs(highest_list[timestamp] - x))
        position_list2 = my_list2.index(closest_value_list2)
        position_timestamp_list2.append(position_list2)
        timestamp_list2.append(closest_value_list2)

    result_position_timestamp_list1.extend(position_timestamp_list1)
    result_position_timestamp_list2.extend(position_timestamp_list2)
    result_timestamp_list1.extend(timestamp_list1)
    result_timestamp_list2.extend(timestamp_list2)

    return (result_position_timestamp_list1, result_timestamp_list1,
            result_position_timestamp_list2, result_timestamp_list2)