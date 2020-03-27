% Run this file before using DL_HSI_denoise function.
% It will check and install the required toolboxes:
%  - ksvdbox and ompbox : for Dictionary Learning
%  - tensorlab : for tensor operations
% and add the corresponding paths to the path list.

fprintf('\n\n****************** SETUP: Toolboxes installation ******************');
fprintf('\n\n The following toolboxes will be downloaded (or attempted to):\n');
fprintf('\n   OMPbox version 10');
fprintf('\n   KSVDbox version 13');
fprintf('\n\n You may also intall these toolboxes manually, if you wish. But');
fprintf('\n remember to add the corresponding paths (using addpath command).');
fprintf('\n\n IMPORTANT: You must have an internet connection.');
fprintf('\n\n IMPORTANT: To successfully install the toolboxes');
fprintf('\n you will need to have MEX setup to compile C files.');
fprintf('\n\n If this is not already setup, please type "n" to exit and then ');
fprintf('\n run "mex -setup" or type "help mex" in the MATLAB command prompt.');
fprintf('\n\n*******************************************************************\n');

install_ack = input('\n\n Do you wish to continue: (y/n)? ','s');

if strcmp(install_ack,'"n"'), return; end

%% Downloading OMP and KSVD Toolboxes if necessary
FS=filesep;
%pathstr = pwd;
%addpath([pathstr,FS,'misc']);
%addpath([pathstr,FS,'toolbox']);

if exist([pwd,FS,'toolbox'],'dir'), cd('toolbox'), end
toolbox_path = pwd;

if ~exist('ksvdbox','dir')

    fprintf('\n\n ******************************************************************');
    fprintf('\n Initialising OMPbox and KSVDBox Setup (download) \n');
    
    try
        % OMPBOX
        if exist([toolbox_path, FS, 'ompbox10.zip'],'file')
            omp_zip=[toolbox_path, FS, 'ompbox10.zip'];
        else %Download ompbox from internet
            omp_zip='http://www.cs.technion.ac.il/%7Eronrubin/Software/ompbox10.zip';
            fprintf('\n Downloading OMP toolbox, please be patient\n');
        end
        unzip(omp_zip,[toolbox_path, FS, 'ompbox']);
               
        % KSVDBOX
        if exist([toolbox_path, FS, 'ksvdbox13.zip'],'file')
            KSVD_zip=[toolbox_path, FS, 'ksvdbox13.zip'];
        else %Download ompbox from internet
            KSVD_zip='http://www.cs.technion.ac.il/%7Eronrubin/Software/ksvdbox13.zip';
            fprintf('\n Downloading KSVD toolbox, please be patient\n');
        end
        unzip(KSVD_zip,[toolbox_path, FS, 'ksvdbox']);

        fprintf('\n KSVDBox and OMPBox Installation Successful');
        fprintf('\n ******************************************************************\n');
        
    catch e
        fprintf(1,'\n There was an error! The message was:\n%s',e.message);
        fprintf(1,'\n The identifier was:\n%s',e.identifier);

        fprintf('\n\n !! KSVDBox and OMPBox Installation Failed. Please try and download them manually.\n');
        cd(toolbox_path);
    end
end

%% Compiling mex files in OMP and KSVD Toolboxes if necessary

if isempty(dir(['ksvdbox',FS,'private',FS,'*.mex*']))

    fprintf('\n ******************************************************************');
    fprintf('\n Compiling OMPbox and KSVDBox \n');
    
    try
        % OMPBOX
        cd([toolbox_path, FS, 'ompbox', FS, 'private']);
        make;
        
        % KSVDBOX
        cd([toolbox_path, FS, 'ksvdbox', FS, 'private']);
        make;
        
        cd(toolbox_path);
        fprintf('\n KSVDBox and OMPBox Compilation Successful');
        fprintf('\n ******************************************************************\n');
             
    catch e
        fprintf(1,'\n There was an error! The message was:\n%s',e.message);
        fprintf(1,'\n The identifier was:\n%s',e.identifier);

        fprintf('\n\n !! KSVDBox and OMPBox Compilation Failed. Please try and compile them manually.\n');
        cd(toolbox_path);
    end
end

%% Downloading TENSORLAB toolbox if necessary
if ~exist('tensorlab_2016-03-28','dir')

    fprintf('\n ******************************************************************');
    fprintf('\n Initialising Tensorlab Setup');
    
    try
        if exist([toolbox_path, FS, 'tensorlab_2016-03-28.zip'],'file')
            tensorlab_zip=[toolbox_path, FS, 'ompbox10.zip'];
        else %Download tensorbox from internet
            tensorlab_zip='https://www.tensorlab.net/download.php?t=1585224153&k=259e93b6&e=sXub4M-rQ144P4Q6-vPbPQ';
            fprintf('\n\n Downloading Tensorlab toolbox, please be patient\n\n');
        end
        unzip(tensorlab_zip,[toolbox_path, FS, 'tensorlab_2016-03-28']);
        
        fprintf('\n Tensorlab Installation Successful');
        fprintf('\n ******************************************************************\n');
        
    catch e
        fprintf(1,'\n There was an error! The message was:\n%s',e.message);
        fprintf(1,'\n The identifier was:\n%s',e.identifier);

        fprintf('\n\n !! Tensorlab Installation Failed. Please try and download it manually.\n');
        cd(toolbox_path);
    end
end

%% Copying files to ksvdbox folder
% So they can use the functions in private folder
if ~exist([toolbox_path, FS, 'ksvdbox/image_denoise_lr.m'],'file')
    movefile(['..', FS,'core', FS,'image_denoise_lr.m'],[toolbox_path, FS, 'ksvdbox']);
    fprintf('\n Moving image_denoise_lr.m file to ksvdbox folder.\n');
    
    movefile(['..', FS,'misc', FS,'image_denoise_various_noise.m'],[toolbox_path, FS, 'ksvdbox']);
    fprintf(' Moving image_denoise_various_noise.m file to ksvdbox folder.\n');
    
    movefile(['..', FS,'misc', FS,'ksvd_various_noise.m'],[toolbox_path, FS, 'ksvdbox']);
    fprintf(' Moving ksvd_various_noise.m file to ksvdbox folder.\n');
    
    movefile(['..', FS,'misc', FS,'ompdenoise2_various_noise.m'],[toolbox_path, FS, 'ksvdbox']);
    fprintf(' Moving ompdenoise2_various_noise.m file to ksvdbox folder.\n');
end

addpath(genpath(toolbox_path));