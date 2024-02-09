%% Load .mat file
% filename = 'dataStruct_24';
% mat_file = 'dataStruct.mat';
% load(matfile, filename);

[filenames, pathname] = uigetfile('.mat');
load(fullfile(pathname, filenames));


%% divide into gait cycles considering RIGHT foot
    
% Define filter parameters
order = 2;              % Filter order
cutoff_frequency = 2;   % Cutoff frequency in Hz
sampling_rate = 60;     % Sample rate in Hz

% Calculate normalized cutoff frequency (Nyquist frequency)
normalized_cutoff = cutoff_frequency / (sampling_rate / 2);

% Design the Butterworth filter
[b,a] = butter(order, normalized_cutoff);

frame = dataStruct_24.frame;
% heel strike
right_heel = dataStruct_24.footContact(3).footContacts; 				% CERTO

% toe-off
right_toe = dataStruct_24.footContact(4).footContacts; 				% CERTO
    
% foot position
foot = dataStruct_24.segmentData(18).position(:,end);					% CERTO	
foot = filtfilt(b, a, foot);
norm_foot = (foot - min(foot)) / ( max(foot) - min(foot) );    

% toe position
toe = dataStruct_24.segmentData(19).position(:,end);					% CERTO
toe = filtfilt(b, a, toe);
norm_toe = (toe - min(toe)) / ( max(toe) - min(toe) );

% plot data
plot(frame, right_heel); hold on;
plot(frame, norm_foot, 'k');
xlabel('Sample');
ylabel('Foot contact');
title('Right Heel');
ylim([-0.5 1.5]); xlim([1 length(right_heel)]);

txt = input("\nDo you want manual edit? Write 'y' for yes, 'n' for no: ","s");
if (txt == "Y") || (txt == "y")    
    close all;
    edit = true;
    
    while(edit)
        % plot foot position
        plot(foot); title('Foot');
        
        thr = input('Introduce threshold: ');
        close all;
        foot_contact = [];
    
        for i = 1:length(foot)
            if foot(i) <= thr
                foot_contact(end+1,1) = 1;
            else
                foot_contact(end+1,1) = 0;
            end
        end

        % plot again all data to verify if events detected correctly
        plot(frame, foot_contact); hold on;
        plot(frame,norm_foot);
        xlabel('Sample');
        ylabel('Foot contact');
        title('Right Heel');
        ylim([-0.5 1.5]); xlim([1 length(foot_contact)]);

        txt = input("Change threshold? Y/N ","s");
        if (txt == "N") || (txt == "n")
            dataStruct_24.HS_RThr = thr; % CERTO
            dataStruct_24.footContact(3).footContacts = foot_contact;					% CERTO
            edit = false;
        else
            close all;
        end
    end
end
    close all;        
    edit = true;
    
    % edit toe contact
    while(edit)
        plot(toe); title('Toe');
                
        thr = input('Introduce threshold: ');
        close all;
        toe_contact = [];
        for i = 1:length(toe)
            if toe(i) <= thr
                toe_contact(end+1,1) = 1;
            else
                toe_contact(end+1,1) = 0;
            end
        end

        plot(frame, toe_contact); hold on;
        plot(frame, norm_toe, 'k');
        legend('Normalized Toe Position','Contact');

        txt = input("Change threshold? Y/N ","s");
        if (txt == "N") || (txt == "n")
            dataStruct_24.TO_RThr = thr; % CERTO
            dataStruct_24.footContact(4).footContacts = toe_contact;					% CERTO
            edit = false;
        else
            close all;
        end
    end

close all;

%% divide into gait cycles considering LEFT foot
    
% Define filter parameters
order = 2;              % Filter order
cutoff_frequency = 2;   % Cutoff frequency in Hz
sampling_rate = 60;     % Sample rate in Hz

% Calculate normalized cutoff frequency (Nyquist frequency)
normalized_cutoff = cutoff_frequency / (sampling_rate / 2);

% Design the Butterworth filter
[b,a] = butter(order, normalized_cutoff);

frame = dataStruct_24.frame;
% heel strike
left_heel = dataStruct_24.footContact(1).footContacts; % CERTO

% toe-off
left_toe = dataStruct_24.footContact(2).footContacts; % CERTO
    
% foot position
foot = dataStruct_24.segmentData(22).position(:,end); % CERTO
foot = filtfilt(b, a, foot);
norm_foot = (foot - min(foot)) / ( max(foot) - min(foot) );    

% toe position
toe = dataStruct_24.segmentData(23).position(:,end); % CERTO
toe = filtfilt(b, a, toe); 
norm_toe = (toe - min(toe)) / ( max(toe) - min(toe) );

% plot data
plot(frame, left_heel); hold on;
plot(frame, norm_foot, 'k');
xlabel('Sample');
ylabel('Foot contact');
title('Left Heel');
ylim([-0.5 1.5]); xlim([1 length(left_heel)]);

txt = input("\nDo you want manual edit? Write 'y' for yes, 'n' for no: ","s");
if (txt == "Y") || (txt == "y")    
    close all;
    edit = true;
    
    while(edit)
        % plot foot position
        plot(foot); title('Foot');
        
        thr = input('Introduce threshold: ');
        close all;
        foot_contact = [];
    
        for i = 1:length(foot)
            if foot(i) <= thr
                foot_contact(end+1,1) = 1;
            else
                foot_contact(end+1,1) = 0;
            end
        end

        % plot again all data to verify if events detected correctly
        plot(frame, foot_contact); hold on;
        plot(frame,norm_foot);
        xlabel('Sample');
        ylabel('Foot contact');
        title('Left Heel');
        ylim([-0.5 1.5]); xlim([1 length(foot_contact)]);

        txt = input("Change threshold? Y/N ","s");
        if (txt == "N") || (txt == "n")
            dataStruct_24.HS_LThr = thr; % CERTO
            dataStruct_24.footContact(1).footContacts = foot_contact; % CERTO
            edit = false;
        else
            close all;
        end
    end

    close all;        
    edit = true;
    
    % edit toe contact
    while(edit)
        plot(toe); title('Toe');
                
        thr = input('Introduce threshold: ');
        close all;
        toe_contact = [];
        for i = 1:length(toe)
            if toe(i) <= thr
                toe_contact(end+1,1) = 1;
            else
                toe_contact(end+1,1) = 0;
            end
        end

        plot(frame, toe_contact); hold on;
        plot(frame, norm_toe, 'k');
        legend('Normalized Toe Position','Contact');

        txt = input("Change threshold? Y/N ","s");
        if (txt == "N") || (txt == "n")
            dataStruct_24.TO_LThr = thr; % CERTO
            dataStruct_24.footContact(2).footContacts = toe_contact; % CERTO
          
            edit = false;
        else
            close all;
        end
    end
end

close all;

%% create final labeling
right_FC = dataStruct_24.footContact(3).footContacts | dataStruct_24.footContact(4).footContacts;
left_FC = dataStruct_24.footContact(1).footContacts | dataStruct_24.footContact(2).footContacts;
dataStruct_24.labeling = right_FC - left_FC;
plot(dataStruct_24.frame, dataStruct_24.labeling); hold on;
xlabel('Frames');
ylabel('Foot Contact');
title('Right FC + (- Left FC)');

%% save struct into .mat

for i=1:24
    struct_name = sprintf('dataStruct_%d', i);
    save('C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\LABELING_CODE\dataStructs\dataStruct_part_15.mat', struct_name, '-append');
end

% decision = input("Do you rerun the program? Y/N ", "s");
% if decision == "y" || decision == "Y"
%     main_to_analysis
% else
%     disp('Stop.');
% end

%% save everything into the specific folders
main_path = 'C:\Users\diman\OneDrive\Ambiente de Trabalho\DATASET\LABELS_PARTICIPANTS';
 
for subFolderIndex = 1:24
    folder_name = sprintf('Participant_15\\Trial_%d', subFolderIndex);
    folder_path = fullfile(main_path, folder_name);
    struct_name = sprintf('dataStruct_%d', subFolderIndex);

    % Create folder if it doesn't exist
    if ~exist(folder_path, 'dir')
        mkdir(folder_path);
    end
        
    % Save Excel files
    excelFileName = 'sum_FC.csv';
    excelFullPath = fullfile(folder_path, excelFileName);
    writematrix(eval([struct_name, '.labeling']), excelFullPath);
    
    % Save MAT files
    matFileName = sprintf('%s.mat', struct_name);
    matFullPath = fullfile(folder_path, matFileName);
    save(matFullPath, struct_name);
end

%% Define sequences of frames
number_trials = 24;
dataStructCellArray = cell(1, number_trials);
step_frames_array = {};
min_trial_value = [];
result_frames = cell(1, number_trials);
for i = 1:number_trials
    fieldName = ['dataStruct_', num2str(i)];
    dataStructCellArray{i} = evalin('base', fieldName);
end

for trial = 1:number_trials
    current_struct = dataStructCellArray{1, trial};
    struct_length = length(current_struct.labeling);
    
    positions_array = [];
    position_trans_1 = 0;

    trial_frames = [];

    for label = 1:struct_length - 1
        % Check if label is within valid range
        if label + 1 <= struct_length
            current_label = current_struct.labeling(label, 1);
            next_label = current_struct.labeling(label + 1, 1);

            if  current_label ~= next_label
                position_trans = label;
                positions_array = [positions_array, position_trans];
            end          
        end
    end

    for value = 2:length(positions_array)
        frames = positions_array(1, value)   - positions_array(1, value - 1);
        trial_frames = [trial_frames, frames];
    end
    min_trial_value = [min_trial_value, min(trial_frames)];
    result_frames{trial} = trial_frames;
end

disp(min_trial_value);

%% Load Xsens files (trials)
% Open a dialog to select multiple files
[filenames, pathname] = uigetfile('.mvnx', 'Select one or more files', 'MultiSelect', 'off');
dataStruct_24 = struct;

% Check if the user clicked Cancel
if isequal(filenames, 0)
    disp('User canceled the operation');
else
    selectedFilePath = fullfile(pathname, filenames);
    disp(['Selected file: ' selectedFilePath]);
    tree = load_mvnx(selectedFilePath);

    time = [];
    filename = selectedFilePath;

    for j = 1:length(tree.frame)
        time(end+1) = str2double(tree.frame(j).time)/1000;           % from ms to sec
    end
    time = transpose(time);
    frame = transpose(1:length(tree.frame));

    dataStruct_24.time = time;
    dataStruct_24.frame = frame;

    dataStruct_24.jointAngles = tree.jointData;

    dataStruct_24.segmentData = tree.segmentData;

    dataStruct_24.footContact = tree.footContact;

    dataStruct_24.filename = selectedFilePath;
end

