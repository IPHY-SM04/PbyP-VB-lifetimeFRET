function [Alpha,Norm] = forward_algo_VBEM(A_lase,B_lase,Pi_mat)

% Forward Algorithm: Given observation sequence & model parameters `A_lase`, `B_lase`, `Pi_mat`
% `A_lase`: State transition matrix of size sta_num^2 ¡Á T  
% `B_lase`: Emission probability matrix of size sta_num ¡Á 2  
% `Pi_mat`: Initial state probability vector of size 1 ¡Á sta_num  



N     = size(B_lase,2);
Alpha = zeros(size(A_lase,1),N);               
Norm  = zeros(1,N);

% initialization
Alpha(:,1) = Pi_mat'.*B_lase(:,1); 
Norm(1)    = sum(Alpha(:,1));
Alpha(:,1) = Alpha(:,1)/Norm(1);

% iteration
for n=1:N-1
    Alpha(:,n+1)=(Alpha(:,n)'*A_lase(:,:,n))'.*B_lase(:,n+1); 
    Norm(n+1)=sum(Alpha(:,n+1));
    Alpha(:,n+1)=Alpha(:,n+1)/Norm(n+1);      % normalization
end
end






























