function [D,D_structured,normVec, X] = HO_SuKro_DL_ALS(params)
%HO-SuKro (Higher-order Sum of Kroneckers) Dictionary Learning.
%  [D,D_structured,X] = HO_SuKro_DL(params) runs the dictionary 
%  training algorithm on the specified set of signals (as a field of the 
%  input variable 'params'), returning the trained dictionary 'D', its
%  version before column normalization 'D_structured'
%  and the sparse signal representation matrix 'X'.
%
%  The training algorithm performs an alternate minimization on the
%  variables 'D' (dictionary update step) and 'X' (sparse coding step).
%
%  1) Dictionary update step
%     A projected gradient descent algorithm.
%     Uses the CPD (from tensorlab) as a projection operator on the space
%     of rank-K tensors.
%
%  2) Sparse coding step
%     Two modes of operation for the sparse reconstruction step are
%     available:
%
%     - Sparsity-based minimization, the optimization problem is given by
%
%         min  |Y-D*X|_F^2      s.t.  |X_i|_0 <= T
%         D,X
%
%       where Y is the set of training signals, X_i is the i-th column of
%       X, and T is the target sparsity. 
%  
%     - Error-based minimization, the optimization problem is given by
%
%         min  |X|_0      s.t.  |Y_i - D*X_i|_2 <= EPSILON
%         D,X
%
%       where Y_i is the i-th training signal, EPSILON is the target error.
%
%  --------------------------
%
%  The input arguments organization in this code is based on the K-SVD code
%  available at:  http://www.cs.technion.ac.il/~ronrubin/software.html
%  References:
%  [1] M. Aharon, M. Elad, and A.M. Bruckstein, "The K-SVD: An Algorithm
%      for Designing of Overcomplete Dictionaries for Sparse
%      Representation", the IEEE Trans. On Signal Processing, Vol. 54, no.
%      11, pp. 4311-4322, November 2006.
%  [2] M. Elad, R. Rubinstein, and M. Zibulevsky, "Efficient Implementation
%      of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit",
%      Technical Report - CS, Technion, April 2008.
%
%  Required fields in PARAMS:
%  --------------------------
%
%    'data' - Training data.
%      A matrix containing the training signals as its columns.
%
%    'Tdata' / 'Edata' - Sparse coding target.
%      Specifies the number of coefficients (Tdata) or the target error in
%      L2-norm (Edata) for coding each signal. If only one is present, that
%      value is used. If both are present, Tdata is used, unless the field
%      'codemode' is specified (below).
%
%    'initdict' - Specifies the initial dictionary for the training. It
%      should be a matrix of size nxm, where n=size(data,1).
%
%
%  Optional fields in PARAMS:
%  --------------------------
%
%    'iternum' - Number of training iterations.
%      If not specified, the default is 10.
%
%    'alpha' - Controls the penalization on the displacement rank.
%      It determines the regularization parameter lambda as follows:
%      
%        lambda = params.alpha*norm(Y);
%
%    'memusage' - Memory usage.
%      This parameter controls memory usage of the function. 'memusage'
%      should be one of the strings 'high', 'normal' (default) or 'low'.
%      When set to 'high', the fastest implementation of OMP is used, which
%      involves precomputing both G=D'*D and DtX=D'*X. This increasese
%      speed but also requires a significant amount of memory. When set to
%      'normal', only the matrix G is precomputed, which requires much less
%      memory but slightly decreases performance. Finally, when set to
%      'low', neither matrix is precomputed. This should only be used when
%      the trained dictionary is highly redundant and memory resources are
%      very low, as this will dramatically increase runtime. See function
%      OMP for more details.
%
%    'codemode' - Sparse-coding target mode.
%      Specifies whether the 'Tdata' or 'Edata' fields should be used for
%      the sparse-coding stopping criterion. This is useful when both
%      fields are present in PARAMS. 'codemode' should be one of the
%      strings 'sparsity' or 'error'. If it is not present, and both fields
%      are specified, sparsity-based coding takes place.
%
%
%  Optional fields in PARAMS - advanced:
%  -------------------------------------
%
%    'maxatoms' - Maximal number of atoms in signal representation.
%      When error-based sparse coding is used, this parameter can be used
%      to specify a hard limit on the number of atoms in each signal
%      representation (see parameter 'maxatoms' in OMP2 for more details).
%
%
%   Summary of all fields in PARAMS:
%   --------------------------------
%
%   Required:
%     'data'                   training data
%     'Tdata' / 'Edata'        sparse-coding target
%     'initdict'               initial dictionary / dictionary size
%
%   Optional (default values in parentheses):
%     'iternum'                number of training iterations (10)
%     'memusage'               'low, 'normal' or 'high' ('normal')
%     'codemode'               'sparsity' or 'error' ('sparsity')
%     'maxatoms'               max # of atoms in error sparse-coding (none)
%
%
%  Reference:
%  [3] C.F. Dantas, M. N. da Costa, R.R. Lopes, "Learning Dictionaries as a
%      sum of Kronecker products", To appear.



%% Required parameters

if (isfield(params,'initdict'))
    D = params.initdict;
else
%     error('TO-IMPLEMENT!: Take random data samples as initial dictionary. Initial dictionary should be provided in field params.initdict');
    % Initialize dictionary with data samples chosen randomly.
    perm = randperm(size(params.data,2));
    D = params.data(:,perm(1:params.dictsize));
    normCols = sqrt(sum(D.^2));
    
    % Replace zero-norm columns
    zeronormCols = find(normCols < 1e-6);
    idx = params.dictsize;
    while any(zeronormCols)
        for kCol = 1:length(zeronormCols)
            idx = idx +1; % Take next random data
            D(:,zeronormCols(kCol)) = params.data(perm(idx));
            normCols(zeronormCols(kCol)) = norm(D(:,zeronormCols(kCol)));
        end
        zeronormCols = find(normCols < 1e-6);
    end
end

if (isfield(params,'data'))
    Y = params.data;
else
    error('Training data should be provided in field params.data');
end


if (size(Y,2) < size(D,2))
  error('Number of training signals is smaller than number of atoms to train');
end


%% Parameter setting

% iteration count %

if (isfield(params,'iternum'))
  iternum = params.iternum;
else
  iternum = 10;
end

% Indices for the reordering operators
% It takes too much memory
% [N,M] = size(D);
% Ni = sqrt(N);
% Mi = sqrt(M);
% Pnm = permut(Ni,Mi);
% P_rank = kron(eye(Mi),kron(Pnm,eye(Ni)));
% idx = find(P_rank.'~=0) - (0:(N*M-1)).'*(N*M);
% idx_inv = find(P_rank~=0) - (0:(N*M-1)).'*(N*M);
% clear P_rank Pnm
% Demands less memory
n = params.kro_dims.N;
m = params.kro_dims.M;

data_dims = size(Y,2);
N = data_dims(end);

I = length(n); % nb modes
R = params.alpha; % nb kronecker summing terms

% Initialization of the factors D_ip and normVec
D_ip =  cell(length(n),params.alpha);
normVec = ones(1,prod(m));
if isfield(params,'odct_factors')
    % First summing term correspond to ODCT
    D_ip(:,1) = params.odct_factors;
    % other factors are random
    for i = 1:I
        for p = 2:R
            D_ip{i,p} = randn(n(i),m(i));
            D_ip{i,p} = normc(D_ip{i,p});
        end
    end
elseif isfield(params,'initdict_struct') % Initializing with a given HO-SuKro
    D_ip = params.initdict_struct;
    normVec = params.normVec;
else % random initialization
    for i = 1:I
        for p = 1:R
            D_ip{i,p} = randn(n(i),m(i));
            D_ip{i,p} = normc(D_ip{i,p});
        end
    end    
end


%% Sparse Coding parameter setting
CODE_SPARSITY = 1;
CODE_ERROR = 2;

MEM_LOW = 1;
MEM_NORMAL = 2;
MEM_HIGH = 3;

%%%%% parse input parameters %%%%%

ompparams = {'checkdict','off'};

% coding mode %

if (isfield(params,'codemode'))
  switch lower(params.codemode)
    case 'sparsity'
      codemode = CODE_SPARSITY;
      thresh = params.Tdata;
    case 'error'
      codemode = CODE_ERROR;
      thresh = params.Edata;
    otherwise
      error('Invalid coding mode specified');
  end
elseif (isfield(params,'Tdata'))
  codemode = CODE_SPARSITY;
  thresh = params.Tdata;
elseif (isfield(params,'Edata'))
  codemode = CODE_ERROR;
  thresh = params.Edata;

else
  error('Data sparse-coding target not specified');
end


% max number of atoms %

if (codemode==CODE_ERROR && isfield(params,'maxatoms'))
  ompparams{end+1} = 'maxatoms';
  ompparams{end+1} = params.maxatoms;
end


% memory usage %

if (isfield(params,'memusage'))
  switch lower(params.memusage)
    case 'low'
      memusage = MEM_LOW;
    case 'normal'
      memusage = MEM_NORMAL;
    case 'high'
      memusage = MEM_HIGH;
    otherwise
      error('Invalid memory usage mode');
  end
else
  memusage = MEM_NORMAL;
end

% omp function %

if (codemode == CODE_SPARSITY)
  ompfunc = @omp;
else
  ompfunc = @omp2;
end

if (I == 2)
  ompfunc_mine = @SolveOMP_2D;
else
  ompfunc_mine = @SolveOMP_tensor;
end  

% data norms %

XtX = []; XtXg = [];
if (codemode==CODE_ERROR && memusage==MEM_HIGH)
  XtX = sum(Y.^2);
end

err = zeros(1,iternum);
gerr = zeros(1,iternum);

if (codemode == CODE_SPARSITY)
  errstr = 'RMSE';
else
  errstr = 'mean atomnum';
end

%% Sparse Coding
% normVec = ones(1,prod(m));
if isfield(params, 'my_omp_training') && params.my_omp_training
    % Using my homemade OMP functions for fair comparison (Cassio)
    % Gives same result as the original implementation below. Verified!
    X = zeros(size(D,2),size(Y,2));
    for k_OMP = 1:size(Y,2)
        % Structure OMP
        if isfield(params,'odct_factors') % uses only first summing term of dictionary
%             [X(:,k_OMP),~,~,~] = SolveOMP_tensor(D, Y(:,k_OMP), params.dictsize, D_ip(:,1),normVec.',params.blocksize,1, prod(params.blocksize),0,0,0,thresh);
            [X(:,k_OMP),~,~,~] = ompfunc_mine(D, Y(:,k_OMP), params.dictsize, D_ip(:,1),normVec.',params.blocksize,1, prod(params.blocksize),0,0,0,thresh);
        else
%             [X(:,k_OMP),~,~,~] = SolveOMP_tensor(D, Y(:,k_OMP), params.dictsize, D_ip,normVec.',params.blocksize,params.alpha, prod(params.blocksize),0,0,0,thresh);
            [X(:,k_OMP),~,~,~] = ompfunc_mine(D, Y(:,k_OMP), params.dictsize, D_ip,normVec.',params.blocksize,params.alpha, prod(params.blocksize),0,0,0,thresh);
        end
%         [X(:,k_OMP),~,~,~] = SolveOMP(D, Y(:,k_OMP), params.dictsize, prod(params.blocksize),0,0,0,thresh);
    end
else % Original implementation
    G = [];
    if (memusage >= MEM_NORMAL)
        G = D'*D;
    end

    if (memusage < MEM_HIGH)
      X = ompfunc(D,Y,G,thresh,ompparams{:});

    else  % memusage is high

      if (codemode == CODE_SPARSITY)
        X = ompfunc(D'*Y,G,thresh,ompparams{:});

      else
        X = ompfunc(D'*Y,XtX,G,thresh,ompparams{:});
      end

    end
end
%%%%%%%%%%%%%%%%%%

%% Optimization %%
%%%%%%%%%%%%%%%%%%

%% Alternating Optimization
fprintf('Iteration:              ')
    
obj_global = [];
verbose=false;

tic
for k = 1:iternum
    fprintf(repmat('\b',1,13));
    fprintf('%4d / %4d  ',k,iternum);

    %% Dictionary update step - ALS
    %TODO avoid duplicating X and Y and using full(X)
    disp('')
    X_matrix  = X;
    X  = reshape(full(X),[m N]);
%     X  = reshape(full(diag(normVec)*X),[m N]); % Normalizing X
    Y = reshape(params.data, [n N]);
    
%     profile on
    DictUpdateALS2
%     tic, DictUpdateALS2_2D, toc
%     profile off
%     profsave(profile('info'),'myprofile_results')
    
    Y = params.data;
    X = X_matrix;
    
%     obj_global = [obj_global obj];

    % Dict_improvement = norm(normc(D_hat) - D)
    
    D_structured = zeros(size(D));
    for p = 1:R 
        % Normalize each factor?
%         for i=1:I
%             D_ip{i,p} = normc(D_ip{i,p});
%         end
        D_structured = D_structured + kron(D_ip(I:-1:1,p)); %TODO get rid of kron. Use directly the D_ip
    end
    
    % Dictionary column normalization
%     normVec = normColsKron(D_ip,m); % TO COMPLETE.
    
    normVec = sqrt(sum(D_structured.^2));
%     D = normc(D_structured);
    D = D_structured./repmat(normVec,size(D,1),1);
    D_structured = D_ip;

    %% Sparse coding step
    if isfield(params, 'my_omp_training') && params.my_omp_training
        % Using my homemade OMP functions
        for k_OMP = 1:size(Y,2)
%             [X(:,k_OMP),~,~,~] = SolveOMP_tensor(D, Y(:,k_OMP), params.dictsize, D_ip,normVec.',params.blocksize,params.alpha, prod(params.blocksize),0,0,0,thresh);
            [X(:,k_OMP),~,~,~] = ompfunc_mine(D, Y(:,k_OMP), params.dictsize, D_ip,normVec.',params.blocksize,params.alpha, prod(params.blocksize),0,0,0,thresh);
        end
    else % Original implementation
        G = [];
        if (memusage >= MEM_NORMAL)
            G = D'*D;
        end

        if (memusage < MEM_HIGH)
          X = ompfunc(D,Y,G,thresh,ompparams{:});

        else  % memusage is high

          if (codemode == CODE_SPARSITY)
            X = ompfunc(D'*Y,G,thresh,ompparams{:});

          else
            X = ompfunc(D'*Y,XtX,G,thresh,ompparams{:});
          end

        end
    end
end
total_time = toc; 
if verbose, fprintf('    Elapsed time: %.1fs\n',total_time); end
   

end
