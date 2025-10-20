function [log_P,KL_Pi,KL_k,KL_kappa,KL_lamda] = log_P_XofstaNum_FLIM(log_Norm_qz,q_k,q_lamda,W_kappa,W_Pi,u_Pi,u_kappa)
staNum_1 = length(W_Pi);
staNum = size(q_lamda,1)/2;
% log_P_1=0;
% if staNum==2
%     for i=1:staNum
%         log_P_1=log_P_1+log_beta_multivar(W_Pi)-log_beta_multivar(u_Pi);
%     end
% else
%     for i=1:staNum
%         W_kappa_temp=W_kappa(i,:);  u_kappa_temp=u_kappa(i,:);
%         W_kappa_temp(i)=[];         u_kappa_temp(i)=[];
%         log_P_1=log_P_1+log_beta_multivar(W_kappa_temp)+log_beta_multivar(W_Pi);
%         log_P_1=log_P_1-log_beta_multivar(u_kappa_temp)-log_beta_multivar(u_Pi);
%     end
% end


% log_P = log_Norm_qz - KL(q_Pi||p(Pi)) - KL(q_k||p(k)) - KL(q_kappa||p(kappa)) - KL(q_lamda||p(lamda))

% KL(q_k||p(k))
x_k = q_k(1:staNum_1,:); q_k = q_k(staNum_1+1:end,:);
F_k = q_k.*(log(q_k)-log(1e-6)-1e-6.*x_k);
KL_k = sum(sum(...
    (x_k(:,2:end)-x_k(:,1:end-1)).*(0.5*(F_k(:,1:end-1)+F_k(:,2:end)))...
    ));

% KL(q_lamda||p(lamda))
x_lamda = q_lamda(1:staNum,:); q_lamda = q_lamda(staNum+1:end,:);
F_lamda = q_lamda.*(log(q_lamda)-x_lamda);
KL_lamda = sum(sum(...
    (x_lamda(:,2:end)-x_lamda(:,1:end-1)).*(0.5*(F_lamda(:,1:end-1)+F_lamda(:,2:end)))...
    ));

% KL(q_Pi||p(Pi))




W_Pi_0=sum(W_Pi);
KL_Pi = log_beta_multivar(u_Pi)-log_beta_multivar(W_Pi) + (W_Pi-u_Pi)*(psi(W_Pi)-psi(W_Pi_0))';

% KL(q_kappa||p(kappa))
W_kappa_0=sum(W_kappa-diag(diag(W_kappa)),2);
if staNum_1==2
    KL_kappa = 0;
else
    KL_kappa = zeros(1,staNum_1);
    for i = 1:staNum_1
        W_1 = W_kappa(i,:);  u_1 = u_kappa(i,:);
        W_1(i) = [];         u_1(i) = [];
        KL_kappa(i) = log_beta_multivar(u_1)-log_beta_multivar(W_1)+(W_1-u_1)*(psi(W_1)-psi(W_kappa_0(i)))';
    end
    KL_kappa = sum(KL_kappa);
end


% KL_psi = KL_k + KL_lamda + KL_Pi + KL_kappa;
log_P = log_Norm_qz - KL_k - KL_lamda - KL_Pi - KL_kappa;
end



