% >>> Alternating Least-Squares for optimizing HO-SuKro terms. <<<
% This version optimizes all rank terms (p=1:R) simultaneously. This
% accelerates the learning process.
%
% Parameters :
% - Sizes of factors D{i,p} is nixmi for any p (stocked in memory)
%       n = [n1 n2 n3]; % size I
%       m = [m1 m2 m3];
% - N : Number of training samples
% - R : Number of Kronecker summing terms
% - I = length(n) : Number of modes

% addpath ../tensorlab_2016-03-28/

%% Optimizing dictionary, given X and Y

% Convergence measures
N_iter = 30; % maximum number of iterations
obj = zeros(1,N_iter);

converged = false;
tol = 1e-1*sqrt(m);
% if k == iternum, tol = 1e-3*sqrt(m); end % Better accuracy on the last iteration %MODIF : it was 1e-3
diff = zeros(I,R); % Frobenius norm of update on each D_ip
k_ALS = 0;

% D_ip_old = cell(length(n),params.alpha);

% for k_ALS = 1:N_iter
while ~converged, k_ALS = k_ALS + 1;
if verbose, fprintf('%4d,',k_ALS); end
for i0 = circshift(1:I,[0,-1])%[2 3 1]
    % Not sure it is necessary - By instead
    Ui0 = cell(R,1); % each Ui0 is (m(i0) x N*prod(m(1:i0-1 i0+1:I)))
    By = cell(1,R);
    for p = 1:R % All indexes
        Ui0{p} = unfold(tmprod(X,D_ip([1:i0-1 i0+1:I],p),[1:i0-1 i0+1:I]),i0); % same as: unfold(X,i0)*kron({eye(N) D_ip{fliplr([1:i0-1 i0+1:I]),p}}).';
        By{p} = -unfold(Y,i0)*Ui0{p}.';
    end
    Ui0 = cell2mat(Ui0);
    By = cell2mat(By);
    
    A = Ui0*Ui0.';
    
    Di0 =  -(A.'\By.').';
%     Di0 = (Ui0.'\unfold(Y,i0).').'; % Gives same result, but slower
    
    for p0 = 1:R
        diff(i0,p0) = norm(D_ip{i0,p0}-Di0(:,m(i0)*(p0-1) + (1:m(i0))),'fro');
%         D_ip_old{i0,p0} = D_ip{i0,p0};
        D_ip{i0,p0} = Di0(:,m(i0)*(p0-1) + (1:m(i0)));
    end
end

% Stop Criterion
if verbose, diff, end
if ( all(mean(diff,2) < tol.') || (k_ALS >= N_iter) )
    converged = true;
    % disp(['Total nº of iterations: ' num2str(iter)]);
end

% Other stop criterion (potentially more costly)
% Calculate the objective function
% Y_r = zeros([n N]);
% for p=1:R
%     Y_r = Y_r + tmprod(X,D_ip(1:I,p),1:I); % Y = D*X, gives sames results as Y = Y + kron(D_ip_oracle(1:I,p))*X(:);
% end
% 
% obj(k_ALS) = norm(Y(:)-Y_r(:),'fro');

% Difference in complete dictionary
% D_structured = zeros(size(D));
% for p = 1:R %TODO get rid of kron. Use directly the D_ip
% %         for i=1:I
% %             D_ip{i,p} = normc(D_ip{i,p});
% %         end
%     D_structured = D_structured + kron(D_ip(I:-1:1,p));
% end
% D_structured_old = zeros(size(D));
% for p = 1:R %TODO get rid of kron. Use directly the D_ip
% %         for i=1:I
% %             D_ip{i,p} = normc(D_ip{i,p});
% %         end
%     D_structured_old = D_structured_old + kron(D_ip_old(I:-1:1,p));
% end
% norm(D_structured-D_structured_old,'fro')

end
