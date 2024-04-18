from create_dataset import *

if __name__ == '__main__':
    NUM_PARTICIPANTS = 15
    TEST_PARTICIPANTS = ['15', '08', '03']
    VAL_PARTICIPANTS = ['02', '11']

    data = Dataset()
    data.split_dataset(NUM_PARTICIPANTS, TEST_PARTICIPANTS, VAL_PARTICIPANTS)
