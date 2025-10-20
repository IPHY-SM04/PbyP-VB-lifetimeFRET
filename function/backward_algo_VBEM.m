function [Beta] = backward_algo_VBEM(A_lase,B_lase)                 

% Backward Algorithm: Given observation sequence & model parameters `A_lase`, `B_lase`, backward probability matrix `Beta`
% `A_lase`: State transition matrix of size sta_num × sta_num × N 
% `B_lase`: emission probability matrix of size sta_num × N 

N       = size(B_lase,2);
sta_num = size(A_lase,1);
Beta    = zeros(sta_num,N);

% initialization
Beta(:,N) = 1/sta_num;

% iteration
for n = N-1:-1:1
    Beta(:,n) = A_lase(:,:,n)*(B_lase(:,n+1).*Beta(:,n+1));
    Beta(:,n) = Beta(:,n)/sum(Beta(:,n));                    % normalization
end
end