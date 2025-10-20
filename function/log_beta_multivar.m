function [log_beta] = log_beta_multivar(W)
% W is 1 * N matrix
% log_beta=log(beta(W(1),...,W(N)))=log(gamma(W(1)));
N=length(W);
W_0 = sum(W);
W_temp = W-floor(W)+1;
log_beta=0;

for i=1:N
    W_temp1=W_temp(i):1:W(i)-1;
    log_beta=log_beta+sum(log(W_temp1))+log(gamma(W_temp(i)));
end
W_0_temp=W_0-floor(W_0)+1:1:W_0-1;
log_beta=log_beta-sum(log(W_0_temp))-log(gamma(W_0_temp(1)));

end

