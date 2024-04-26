from tqdm import tqdm
import os
import zipfile
import re

NUMBER_PARTICIPANTS = 15
NUMBER_TRIALS = 24

path_zip = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\PARTICIPANTS'
contents = os.listdir(path_zip)

for participant in range(NUMBER_PARTICIPANTS):
    with (zipfile.ZipFile(os.path.join(path_zip, contents[participant]), 'r') as zip_ref):
        zip_contents = zip_ref.namelist()

        video_files = [item for item in zip_contents if item.endswith('avi') and item.__contains__('gait')]
        sorted_video_files = sorted(video_files, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))

        for trial in tqdm(range(NUMBER_TRIALS), desc=f"Participant [{participant + 1}] - Processing Trials"):
            zip_ref.extract(sorted_video_files[trial], r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\GAIT_VIDEOS')







