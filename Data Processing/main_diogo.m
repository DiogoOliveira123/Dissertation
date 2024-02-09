%% Load Xsens files (trials)
% % Open a dialog to select multiple files
% [filenames, pathname] = uigetfile('.mvnx', 'Select one or more files', 'MultiSelect', 'off');
% dataStruct = struct;
% 
% % Check if the user cliccked Cancel
% if isequal(filenames, 0)
%     disp('User canceled the operation');
% else
%     selectedFilePath = fullfile(pathname, filenames);
%     disp(['Selected file: ' selectedFilePath]);
%     tree = load_mvnx(selectedFilePath);
% 
%     time = [];
%     filename = selectedFilePath;
% 
%     for j = 1:length(tree.frame)
%         time(end+1) = str2double(tree.frame(j).time)/1000;           % from ms to sec
%     end
%     time = transpose(time);
%     frame = transpose(1:length(tree.frame));
% 
%     dataStruct.time = time;
%     dataStruct.frame = frame;
% 
%     dataStruct.jointAngles = tree.jointData;
% 
%     dataStruct.segmentData = tree.segmentData;
% 
%     dataStruct.footContact = tree.footContact;
% 
%     dataStruct.filename = selectedFilePath;
% end
% clearvars -except dataStruct
% 
% % save tree
% [file,path,indx] = uiputfile('dataStruct.mat');
% if(indx ~= 0)
%     save( fullfile(path, file), 'dataStruct');
% end

%% Load struct
[filenames, pathname] = uigetfile('.mat');
load(fullfile(pathname, filenames));
clearvars -except dataStruct
