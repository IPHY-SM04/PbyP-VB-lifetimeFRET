function [B_mat,lamda_1,log_Norm_q_temp,q] = B_mat_VBEM_FLIM(trace_PbyP,W_lamda,Gamma,IRF,maxTau,resolution,lamda_max)
 
staNum=size(Gamma,1);
lamda_1 =zeros(1,staNum);
delay=trace_PbyP(1,1:end);
xRange1=lamda_max*1.5;
sec_num1=4000;
x1=xRange1/sec_num1:xRange1/sec_num1:xRange1;
lamda = xRange1^(-1)*x1.^2;
x2=(resolution:resolution:maxTau)';
B_mat=zeros(staNum,length(x2));
fun=@(x,k,a1,a2)k/2.*exp(k.*(a1-x)+0.5.*k.^2.*a2.^2).*erfc((a1-x+k.*a2.^2)./(2^0.5*a2));
q = zeros(staNum,length(lamda));     %  q(lamda) = [q_k1; ...; q_ki; ...]

gpuVar_lamda = gpuArray(lamda);
gpuVar_W = gpuArray(W_lamda);
gpuVar_Gamma = gpuArray(Gamma);
gpuVar_delay = gpuArray(delay);
gpuVar_x2 = gpuArray(x2);
gpuVar_IRF = gpuArray(IRF);

gpu_log_q = gpuVar_W(:,1).*log(gpuVar_lamda/2) ...
          + gpuVar_W(:,2).*gpuVar_lamda        ...
          + gpuVar_W(:,3).*gpuVar_lamda.^2     ...
          + gpuVar_Gamma*log(erfc(    (gpuVar_IRF(1)-gpuVar_delay)/(2^0.5*gpuVar_IRF(2))+gpuVar_lamda'*gpuVar_IRF(2)/2^0.5    ))';

gpu_max_log_q = max(gpu_log_q,[],2);
gpu_q_nuNorm = exp(gpu_log_q-gpu_max_log_q);
gpu_AA = 0.5*(gpu_q_nuNorm(:,1:end-1)+gpu_q_nuNorm(:,2:end))*(gpuVar_lamda(2:end)-gpuVar_lamda(1:end-1))';  % AA = [staNum, 1];
gpu_q =  gpu_q_nuNorm./gpu_AA;          q = gather(gpu_q);                         % normalized distribution
gpu_BB = log(fun(gpuVar_x2,gpuVar_lamda,gpuVar_IRF(1),gpuVar_IRF(2)));           % size(y) = [length(x2) length(lamda)]
% gpu_CC = permute(gpu_q,[3 2 1]);
% gpu_Bmat = permute((exp(sum(   0.5*(gpu_CC(:,1:end-1,:).*gpu_BB(:,1:end-1)+gpu_CC(:,2:end,:).*gpu_BB(:,2:end))...
%          .*(gpuVar_lamda(2:end)-gpuVar_lamda(1:end-1)), 2 ))),[3 1 2]);

gpu_Bmat = exp(    0.5*((gpuVar_lamda(2:end)-gpuVar_lamda(1:end-1)).*gpu_q(:,1:end-1)) * gpu_BB(:,1:end-1)'...
                 + 0.5*((gpuVar_lamda(2:end)-gpuVar_lamda(1:end-1)).*gpu_q(:,2:end))   * gpu_BB(:,2:end)' ...
               );

lamda_1 = 0.5*(lamda(2:end)-lamda(1:end-1))*...
    (lamda(1:end-1).*q(:,1:end-1)+lamda(2:end).*q(:,2:end))';
gpu_log_Norm_q_temp = sum(gpu_max_log_q+log(gpu_AA));

B_mat = gather(gpu_Bmat);
log_Norm_q_temp = gather(gpu_log_Norm_q_temp);
q = [lamda; q];

end

