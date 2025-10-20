function [Xi] = P_Xi_VBEM(trace_PbyP,A_lase,B_lase,forward_prob,backward_prob)
% output Xi is sta_num*sta_num*N-1 and Xi(i,j,t) = P(Z(t)=i,Z(t+1)=j|trace_PbyP)


N = length(trace_PbyP);
alpha = permute(forward_prob,[1,3,2]);
beta_B = permute(B_lase.*backward_prob,[3,1,2]);


% Xi = matrix(sta_num,sta_num,N-1);
Xi = (alpha(:,:,1:N-1).*A_lase(:,:,1:N-1)).*beta_B(:,:,2:N);

% normalization
Xi=Xi./sum(sum(Xi));
end
