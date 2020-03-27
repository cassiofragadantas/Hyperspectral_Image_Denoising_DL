function M = unfold(T,m)
% Matricize tensor T w.r.t. mode m
% Equivalent to tens2mat(T,m,[1:m-1 m+1:ndims(T)])

sizes = size(T);
T = permute(T,[m 1:m-1 m+1:ndims(T)]);
% M = reshape(T,sizes(m),prod(sizes([1:m-1 m+1:ndims(T)])));
M = reshape(T,sizes(m),[]);

% T = permute(T,[m 1:m-1 m+1:ndims(T)]);
% M = reshape(T,size(T,1),[]);

end