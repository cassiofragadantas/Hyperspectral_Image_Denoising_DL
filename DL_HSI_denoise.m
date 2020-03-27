function [imout, exec_times] = DL_HSI_denoise(im)
% This is the MAIN SCRIPT of a Hyperspectral Image denoising approach which
% combines low-rankness and sparsity (through Dictionary Learning).
%
%  Reference:
%  [1] C. F. Dantas, J.E. Cohen and R. Gribonval, "Hyperspectral Image 
%        Denoising using Dictionary Learning". WHISPERS 2019. 
%        (Available at: https://hal.inria.fr/hal-02175630v1)
%
%
%  Cassio Fraga Dantas
%  cassiofragadantas@gmail.com
%
%  March 2020

%% Simulation parameters %%
params.algo_type = 2;       % Choose the dictionary type to be used:
                            % 2. HO-SuKro  3. K-SVD  4. ODCT
                            % 1. SuKro (Not included! Older version of HO-SuKro, but slower)
params.iternum = 20;        % Nb. iterations of Dict.update + Sparse coding. Default is 20.
params.trainnum = 20000;    % Number of training samples
                            % Color images (or HSI whispers): [1000 2000 5000 10000 20000];
%Sub-dictionaries sizes
params.blocksize = [6, 6];  % 2D-patches dimensions (used in Whisper's paper)
blocksize_m = [12, 12];     % nb. atoms of each subdictionary D1, D2 and D3 respectively

single_dictionary = false;  % Denoise all eigen-images using the same dictionary.
                            % This allows to accelerate the algorithm while losing some performance.

params.gain = 1.16;         % Calibrated gain for Whispers paper: 1.16

show_dict = false;
show_images = false;

params.memusage = 'high';   % Memory usage for OMP denoising: 'low', 'normal' or 'high' (see ompdenoise2.m)
params.my_omp_training = false; % option 'true' not implemented in this repository (see HO-SuKro-v2 repository)
params.my_omp_denoise = false;  % option 'true' not implemented in this repository
params.sigma0 = 0; % Unused! vestigial parameter

% Number of summing Kronecker terms in HO-SuKro (unused for KSVD or ODCT)
params.alpha = 3;

% Subdictionaries (Kronecker factors) dimensions
params.dictsize = prod(blocksize_m);
params.kro_dims.N = params.blocksize;
params.kro_dims.M = blocksize_m;


disp(' ');
disp('***************  Dictionary Learning approach for HSI denoising  ****************');
disp('*                                                                               *');
disp('* This function takes a noisy Hyperspectral image as an input and outputs a     *');
disp('* denoised version of the image. A low-rank approximation of the (matricized)   *');
disp('* image is used as a preprocessing, followed by a patch-based dictionary        *');
disp('* learning denoising approach                                                   *');
disp('*                                                                               *');
disp('*********************************************************************************');

% KSVD, OMP and Tensorlab toolboxes are assumed to be already installed    
addpath(genpath([pwd,filesep,'toolbox']));
addpath('misc'); addpath('core');

% Copying files to ksvdbox folder so they can use functions in private folder
if ~exist(['toolbox', filesep, 'ksvdbox', filesep,'image_denoise_lr.m'],'file')
    movefile(['core', filesep,'image_denoise_lr.m'],['toolbox', filesep, 'ksvdbox']); 
    movefile(['misc', filesep,'image_denoise_various_noise.m'],['toolbox', filesep, 'ksvdbox']);
    movefile(['misc', filesep,'ksvd_various_noise.m'],['toolbox', filesep, 'ksvdbox']);
    movefile(['misc', filesep,'ompdenoise2_various_noise.m'],['toolbox', filesep, 'ksvdbox']);
end


%% Run Simulations
% Convert to double type
im = double(im);
size_im = size(im);

%% Pre-processing (LR)
exec_times.total = tic; % measure total execution time
exec_times.svd = tic; % measure low-rank approximation time
max_rank = 30;
[U, S, V] = svd(reshape(im,size_im(1)*size_im(2),size_im(3)),'econ'); U=U(:,1:max_rank);S=S(:,1:max_rank); V=V(:,1:max_rank); % svds is slower than svd 'econ'
exec_times.svd = toc(exec_times.svd);

clear im; %free memory

% ---- Rank selection ----

sing_vals = diag(S);
% Empirical Criterion 1 - gives result similar to oracle idx+5
%     idx = find((sing_vals(1:max_rank-1) - sing_vals(2:max_rank))./sing_vals(2:max_rank)<5e-3,1); %5e-3, 1e-2
% Empirical Criterion 2
idx = find((sing_vals(1:max_rank-1) - sing_vals(max_rank))./sing_vals(max_rank)<3e-2,1); % 3e-2, 5e-2

if isempty(idx), idx = max_rank; end
fprintf('\nRank used for SVD truncation: %d\n',idx)

params.x = reshape(U(:,1:idx),size_im(1),size_im(2),idx);

% ---- Noise level estimation ----

sigmas = sing_vals(max_rank)./(sqrt(size(U,1))*sing_vals(1:idx)).'; % Underestimates at final columns

%% Sparse step
U_out = U(:,1:idx);

params.maxval = 1/(sqrt(size(U,1))); % Chosen after tests calibrating lambda

if single_dictionary
% ---- Denoising all columns (learn single dictionary) ----
 
    params.sigma = sigmas;
    [imout, dict, dict_struct, normVec, exec_times_all] = image_denoise_various_noise(params); % Alg. 1
    
    exec_times.training = exec_times_all.training;
    exec_times.denoising = exec_times_all.denoising;
    
    U_out = reshape(imout,size_im(1)*size_im(2),idx);

else
% ---- Denoising column by column ----
    
    exec_times.training = zeros(1,idx);
    exec_times.denoising = zeros(1,idx);

    % Skip first columns which already have a high SNR. This can accelerate the
    % algorithm, without compromising significantly the result (not used by default).
%     start_idx = find(sqrt(size(U,1))*sigmas>1e-2,1); % Index of first column below 40dB SNR
%     U_out(:,1:start_idx-1) = U(:,1:start_idx-1); % High-SNR columns are not denoised
    
    start_idx = 1;
    for k_col = start_idx:idx
        fprintf('\n--------- Denoising Column: %d / %d ---------\n',k_col,idx);
        params.x = reshape(U(:,k_col),size_im(1),size_im(2)); % (eigen)image to be denoised   
        params.sigma = sigmas(k_col);

        % ---- Denoising this column ----
        [imout, dict, dict_struct, normVec, exec_times_col] = image_denoise_lr(params); % Alg. 1

        exec_times.training(k_col) = exec_times_col.training;
        exec_times.denoising(k_col) = exec_times_col.denoising;

        U_out(:,k_col) = imout(:);

        % Initialiser prochaine colonne avec precedente
        params.initdict = dict; 
        params.initdict_struct = dict_struct; params.normVec = normVec;
    end
end

imout = U_out*S(1:idx,1:idx)*V(:,1:idx).';
imout = reshape(imout,size_im);

exec_times.total = toc(exec_times.total);


%% Show results (single-run experiment) %%

if(show_dict) % Caution: This might be slow.
    cd toolbox/ksvdbox/private
    for k_atom=1:params.dictsize
        subplot(ceil(sqrt(params.dictsize)),ceil(sqrt(params.dictsize)),k_atom)
        imshow(imnormalize(reshape(dict(:,k_atom),params.blocksize)))
    end
    cd ../../../
end

if (show_images)
    figure; ax = subplot(1,1,1); imshow(im(:,:,1)/params.maxval);
    title(ax,'Input image (first spectral band)'); drawnow

    figure; ax = subplot(1,1,1); imshow(imout(:,:,1)/params.maxval);
    title(ax,'Denoised image (first spectral band)');
end