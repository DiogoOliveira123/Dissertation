import os
import re
import zipfile

directory_participants = 'C:\\Users\\diman\\OneDrive\\Ambiente de Trabalho\\DATASET\\PARTICIPANTS'
contents = os.listdir(directory_participants)

NUMBER_PARTICIPANTS = 15
NUMBER_TRIALS = 24
total_data_stamp = []

for participant in range(1, NUMBER_PARTICIPANTS):
    path_zip = os.path.join(directory_participants, contents[participant])

    """----------- OPEN ZIP FILE -----------"""
    with (zipfile.ZipFile(path_zip, 'r') as zip_ref):
        zip_contents = zip_ref.namelist()

        # All files
        mvnx_files = [item for item in zip_contents if item.endswith('mvnx')]
        mvn_files = [item for item in zip_contents if item.endswith('mvn')]
        stamp_files = [item for item in zip_contents if item.endswith('stamp')]

        sorted_mvnx_files = sorted(mvnx_files, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))
        sorted_mvn_files = sorted(mvn_files, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))
        sorted_stamp_files = sorted(stamp_files, key=lambda x: int(re.search(r'Trial_(\d+)', x).group(1)))

        date_mvnx = []
        date_stamp = []

        """----------- ITERATE THROUGH EVERY TRIAL WITH A PROGRESS BAR -----------"""
        for trial in range(NUMBER_TRIALS):
            zip_info_mvnx = zip_ref.getinfo(sorted_mvnx_files[trial])
            creation_time_mvnx = zip_info_mvnx.date_time
            date_mvnx.append(creation_time_mvnx)

            zip_info_stamp = zip_ref.getinfo(sorted_stamp_files[trial])
            creation_stamp_video = zip_info_stamp.date_time
            date_stamp.append(creation_stamp_video)

        total_data_stamp.extend(date_stamp)

        path_folder = r'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\EXCEL_DATES\date_times_STAMP.txt'

        i = 0
        j = 0

        with open(path_folder, 'w') as txt:
            for date in total_data_stamp:
                txt.write(f'Participant_{i+2} - Trial_{j+1}: {date[2]}/{date[1]}/{date[0]} {date[3]}:{date[4]}:{date[5]}\n')
                j += 1

                if j > 23:
                    i += 1
                    j = 0
                    



