function [y,D,D_not_normalized, normVec, exec_times, nz] = image_denoise_lr(params,msgdelta)
% This function is a variation of image_denoise from ksvdbox used for the
% low-rank experiments. The basic difference is that this there are now two
% input data (previously there was only params.x for the noisy data): 
%
% params.x0: noisy data. Used only to take the average after reconstruction.
% params.x: current denoised data (either after a low-rank pre-processing 
%            or, if this function is called iteratively, last iteration's
%            result). The dictionary is learned over this data.
%
% It is also necessary to have the knowledge of two noise levels
%
% params.sigma0: noise level of the original noisy image (params.x0).
% params.sigma: noise level of params.x (for OMP in the dictionary learning)
% 
%
% This DEMO is based on the image denoising demo available at the KSVDBOX
% package. 
%
%  Reference:
%  [1] M. Elad and M. Aharon, "Image Denoising via Sparse and Redundant
%      representations over Learned Dictionaries", the IEEE Trans. on Image
%      Processing, Vol. 15, no. 12, pp. 3736-3745, December 2006.
%
% The dictionary learning algorithm is exchanged by the SuKro (Sum of Kroneckers)
% dictionary learning technique (sum_separable_dict_learn.m)
%
%  [2] C.F. Dantas, M.N. da Costa and R.R. Lopes, "Learning Dictionaries as
%       a sum of Kronecker products"

%  [Y,D] = IMAGE_DENOISE(PARAMS) denoises the specified (possibly
%  multi-dimensional) signal. Y is the denoised signal and D is
%  the trained dictionary.
%
%  [Y,D] = IMAGE_DENOISE(PARAMS,MSGDELTA) specifies the frequency of message
%  printing during the process. MSGDELTA should be a positive number
%  representing the interval in seconds between messages. A zero or
%  negative value cancels all messages. Default is MSGDELTA=5.
%
%  [Y,D,NZ] = IMAGE_DENOISE(...) also returns the average number of non-zero
%  coefficients in the representations of the denoised blocks.
%
%
%  Required fields in PARAMS:
%  --------------------------
%
%    'x' - Noisy signal.
%      The signal to denoise (can be multi-dimensional). Should be of type
%      double, and (for PSNR computations) with values within [0,1] (to
%      specify a different range, see parameter 'maxval' below).
%
%    'blocksize' - Size of block.
%      Indicates the size of the blocks to operate on. Should be either an
%      array of the form [N1 N2 ... Np], where p is the number of
%      dimensions of x, or simply a scalar N, representing the square block
%      [N N ... N]. See parameter 'stepsize' below to specify the amount of
%      overlap between the blocks.
%
%    'dictsize' - Size of dictionary to train.
%      Specifies the number of dictionary atoms to train by SuKro.
%
%    'psnr' / 'sigma' - Noise power.
%      Specifies the noise power in dB (psnr) or the noise standard
%      deviation (sigma), used to determine the target error for
%      sparse-coding each block. If both fields are present, sigma is used
%      unless the field 'noisemode' is specified (below). When specifying
%      the noise power in psnr, make sure to set the 'maxval' parameter
%      as well (below) if the signal values are not within [0,1].
%
%    'trainnum' - Number of training blocks.
%      Specifies the number of training blocks to extract from the noisy
%      signal for SuKro training.
%
%
%  Optional fields in PARAMS:
%  --------------------------
%
%    'initdict' - Initial dictionary.
%      Specifies the initial dictionary for the SuKro training. Should be
%      either a matrix of size NxL where N=(N1*N2*...*Np), the string
%      'odct' to specify the overcomplete DCT dictionary, or the string
%      'data' to initialize using random signal blocks. When a matrix is
%      specified for 'initdict', L must be >= dictsize, and in this case
%      the dictionary is initialized using the first dictsize columns from
%      initdict. By default, initdict='odct'.
%
%    'stepsize' -  Interval between neighboring blocks.
%      Specifies the interval (in pixels/voxels) between neighboring blocks
%      to denoise in the OMP denoising step. By default, all overlapping
%      blocks are denoised and averaged. This can be changed by specifying
%      an alternate stepsize, as an array of the form [S1 S2 ... Sp] (where
%      p is the number of dimensions of x). This sets the distance between
%      neighboring blocks to be Si in the i-th dimension. Stepsize can also
%      be a scalar S, corresponding to the step size [S S ... S]. Each Si
%      must be >= 1, and, to ensure coverage of the entire noisy signal,
%      size(x,i)-Ni should be a multiple of Si for all i. The default
%      stepsize is 1.
%
%    'iternum' - Number of iterations (N_iter) of the alternating 
%      minimization (iteration = sparse coding + dictionary update).
%      If not specified, the default is 10.
%
%    'maxval' - Maximal intensity value.
%      Specifies the range of the signal values. Used to determine the
%      noise standard deviation when the noise power is specified in psnr.
%      By default, the signal values are assumed to be within [0,1]. When
%      'maxval' is specified, this range changes to [0,maxval].
%
%    'memusage' - Memory usage.
%      This parameter controls memory usage of the function. 'memusage'
%      should be one of the strings 'high', 'normal' (default) or 'low'.
%      When 'memusage' is specified, OMPDENOISE is invoked
%      using this memusage setting. Note that specifying 'low' will
%      significantly increase runtime.
%
%
%  Optional fields in PARAMS - advanced:
%  -------------------------------------
%
%    'noisemode' - Noise power mode.
%      Specifies whether the 'psnr' or 'sigma' fields should be used to
%      determine the noise power. This is useful when both fields are
%      present in PARAMS. 'noisemode' should be one of the string 'psnr' or
%      'sigma'. If it is not present, and both fields are specified,
%      'sigma' is used.
%
%    'gain' - Noise gain.
%      A positive value (usually near 1) controlling the target error for
%      sparse-coding each block. When gain=1, the target error is precisely
%      the value derived from the psnr/sigma fields. When gain is different
%      from 1, the target error is multiplied by this value. The default
%      value is gain = 1.15.
%
%    'lambda' - Weight of the input signal.
%      Specifies the relative weight attributed to the noisy input signal
%      in determining the output. The default value is 0.1*(maxval/sigma),
%      where sigma is the standard deviation of the noise. See function
%      OMPDENOISE for more information.
%
%    'maxatoms' - Maximal number of atoms.
%      This parameter can be used to specify a hard limit on the number of
%      atoms used to sparse-code each block. Default value is
%      prod(blocksize)/2, i.e. half the number of samples in a block. See
%      function OMP2 for more information.
%
%
%   Summary of all fields in PARAMS:
%   --------------------------------
%
%   Required:
%     'x'                      signal to denoise
%     'blocksize'              size of block to process
%     'dictsize'               size of dictionary to train
%     'psnr' / 'sigma'         noise power in dB / standard deviation
%     'trainnum'               number of training signals
%
%   Optional (default values in parentheses):
%     'initdict'               initial dictionary ('odct')
%     'stepsize'               distance between neighboring blocks (1)
%     'iternum'                number of training iterations (10)
%     'maxval'                 maximal intensity value (1)
%     'memusage'               'low, 'normal' or 'high' ('normal')
%     'noisemode'              'psnr' or 'sigma' ('sigma')
%     'gain'                   noise gain (1.15)
%     'lambda'                 weight of input signal (0.1*maxval/sigma)
%     'maxatoms'               max # of atoms per block (prod(blocksize)/2)
%
%
%  References:
%  [1] M. Elad and M. Aharon, "Image Denoising via Sparse and Redundant
%      representations over Learned Dictionaries", the IEEE Trans. on Image
%      Processing, Vol. 15, no. 12, pp. 3736-3745, December 2006.
%
%  See also sum_separable_dict_learn, OMPDENOISE, OMP2.


%  ORIGINAL DEMO BY:
%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  August 2009
%
%  ADAPTED BY:
%  Cassio Fraga Dantas
%  DSPCom Laboratory - Unicamp
%  Campinas, Brasil
%  cassio@decom.fee.unicamp.br
%
%  September 2016

normVec = ones(params.dictsize); % default return value

%%%%% parse input parameters %%%%%

x = params.x;
blocksize = params.blocksize;
trainnum = params.trainnum;
dictsize = params.dictsize;

p = ndims(x);
if (p==2 && any(size(x)==1) && length(blocksize)==1)
  p = 1;
end


% blocksize %
if (numel(blocksize)==1)
  blocksize = ones(1,p)*blocksize;
end


% maxval %
if (isfield(params,'maxval'))
  maxval = params.maxval;
else
  maxval = 1;
  params.maxval = maxval;
end


% gain %
if (isfield(params,'gain'))
  gain = params.gain;
else
  gain = 1.15;
  params.gain = gain;
end


% msgdelta %
if (nargin<2)
  msgdelta = 5;
end

verbose = 't';
if (msgdelta <= 0)
  verbose='';
  msgdelta = -1;
end


% initial dictionary %

if (~isfield(params,'initdict'))
  params.initdict = 'odct';
end

if (isfield(params,'initdict') && ischar(params.initdict))
  if (strcmpi(params.initdict,'odct'))
    params.initdict = odctndict(blocksize,params.kro_dims.M,p);
    % Initialize the kronecker terms D_ip
    I = length(params.kro_dims.M);
    params.odct_factors = cell(I,1);
    for i_idx=1:I
        params.odct_factors{i_idx,1} =  odctdict(params.kro_dims.N(i_idx),params.kro_dims.M(i_idx));
    end

  elseif (strcmpi(params.initdict,'data'))
    params = rmfield(params,'initdict');    % causes initialization using random examples
  else
    error('Invalid initial dictionary specified.');
  end
end

if (isfield(params,'initdict'))
  params.initdict = params.initdict(:,1:dictsize);
end


% noise mode %
if (isfield(params,'noisemode'))
  switch lower(params.noisemode)
    case 'psnr'
      sigma = maxval / 10^(params.psnr/20);
    case 'sigma'
      sigma = params.sigma;
    otherwise
      error('Invalid noise mode specified');
  end
elseif (isfield(params,'sigma'))
  sigma = params.sigma;
elseif (isfield(params,'psnr'))
  sigma = maxval / 10^(params.psnr/20);
else
  error('Noise strength not specified');
end

params.Edata = sqrt(prod(blocksize)) * sigma * gain;   % target error for omp
params.codemode = 'error';

params.sigma = sigma;
params.noisemode = 'sigma';


% make sure test data is not present in params
if (isfield(params,'testdata'))
  params = rmfield(params,'testdata');
end


%%%% create training data %%%

ids = cell(p,1);
if (p==1)
  ids{1} = reggrid(length(x)-blocksize+1, trainnum, 'eqdist');
else
  [ids{:}] = reggrid(size(x)-blocksize+1, trainnum, 'eqdist');
end
params.data = sampgrid(x,blocksize,ids{:});

% remove dc in blocks to conserve memory %
blocksize = 2000;
for i = 1:blocksize:size(params.data,2)
  blockids = i : min(i+blocksize-1,size(params.data,2));
  params.data(:,blockids) = remove_dc(params.data(:,blockids),'columns');
end



%%%%% SuKro Dictionary Learning %%%%%
exec_times.training = tic; % measure training time
if params.algo_type == 1
    if (msgdelta>0)
      disp('Sukro Dictionary training...');
    end
    [D, D_not_normalized] = sum_separable_dict_learn(params);    
elseif params.algo_type == 2
    if (msgdelta>0)
      disp('HO-Sukro Dictionary training...');
    end
    %[D, D_not_normalized] = HO_SuKro_DL(params);
    [D, D_not_normalized, normVec, ~] = HO_SuKro_DL_ALS(params);
elseif params.algo_type == 3
    if (msgdelta>0)
        disp('KSVD training...');
    end
    D = ksvd(params);
    D_not_normalized = D;
else % no training. Use initial dictionary
    if (msgdelta>0)
        disp('No training. Just using the initial dictionary...');
    end
    D = params.initdict;
    D_not_normalized = D;
end
exec_times.training = toc(exec_times.training);

%%%%%  denoise the signal  %%%%%

if (~isfield(params,'lambda'))
%   params.lambda = maxval/(10*sigma); % ORIGINAL VALUE
    params.lambda = maxval/(sigma);  % MODIF! After calibration. Use with left factor (U) denoising OR whole image denoising with LR preprocessing (without LR preprocessing lambda=0 seems better)
    params.lambda0 = maxval/(10*params.sigma0); %MODIF! Should use sigma0
end

params.dict = D;

if (msgdelta>0)
  disp('OMP denoising...');
end

% call the appropriate ompdenoise function
exec_times.denoising = tic; % measure denoising time
if (p==1)
  [y,nz] = ompdenoise1(params,msgdelta);
elseif (p==2)
  % My Modif (Cassio), to use structured OMP
  if isfield(params, 'my_omp_denoise') && params.my_omp_denoise && (params.algo_type==2)
    [y,nz] = ompdenoise2(params,msgdelta,D_not_normalized,normVec);
  else
    [y,nz] = ompdenoise2(params,msgdelta);
  end
elseif (p==3)
  % My Modif (Cassio), to use structured OMP
  if isfield(params, 'my_omp_denoise') && params.my_omp_denoise && (params.algo_type==2)
    [y,nz] = ompdenoise3_lr(params,msgdelta,D_not_normalized,normVec);
  else
    [y,nz] = ompdenoise3_lr(params,msgdelta);
  end
else
  [y,nz] = ompdenoise(params,msgdelta);
end
exec_times.denoising = toc(exec_times.denoising);

end
