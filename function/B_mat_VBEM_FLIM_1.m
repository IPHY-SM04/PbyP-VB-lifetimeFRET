function [B_mat,lamda,log_Norm_q_temp,q] = B_mat_VBEM_FLIM_1(trace_PbyP,W_lamda,Gamma,IRF,maxTau,resolution,lamda_0)

sec_num = 500;
lamda_0 = lamda_0';
staNum = size(Gamma,1);
% lamda_1 =zeros(1,staNum);
delay=trace_PbyP(1,1:end);
x_lamda = zeros(staNum,sec_num+1);       % horizontal ordinate of q(lamda) integral 
q = zeros(staNum,size(x_lamda,2));       % q(lamda) = [q_lamda1; ...; q_lamda_i; ...]
x2=(resolution:resolution:maxTau);      % horizontal ordinate of B_mat
B_mat=zeros(staNum,length(x2));
fun_log_B_mat=@(x,k,a1,a2) log(k)-log(2) + k/2.*(2*(a1-x)+k.*a2.^2) + log(erfc((a1-x+k.*a2.^2)./(2^0.5*a2)));      
fun_log_q_lamdai = @(W,G,t,lamda,IRF,i)W(i,1)*log(lamda/2) + W(i,2)*lamda + W(i,3)*lamda.^2 ...
                                       + Gamma(i,:)*log(erfc((IRF(1)-t+lamda*IRF(2)^2)/(2^0.5*IRF(2))))';
% Find Extreme Values of q(lamda), Newton's method  x_n+1 = x_n - f'(x_n)/f''(x_n)
thr = 1e-7;
for iter = 1:10000
    temp = (IRF(1)-delay+lamda_0*IRF(2)^2)/(2^0.5*IRF(2));
    erfc_temp = erfc(temp);
    temp_2 = exp(-temp.^2)./(pi^0.5*erfc_temp);
    % Newton's method  x_n+1 = x_n - f(x_n)/f'(x_n)
    
    aaa = W_lamda(:,1)./lamda_0 + W_lamda(:,2) + 2*W_lamda(:,3).*lamda_0 - 2^0.5*IRF(2)*sum(Gamma.*temp_2,2);
    bbb =  -W_lamda(:,1)./lamda_0.^2  + 2*W_lamda(:,3) ...
           + sum(     Gamma.*(    2*IRF(2)^2.*(temp).*temp_2 ...
                                - 2*IRF(2)^2.* temp_2.^2    ...
                              )...
                 ,2);   
             
    lamda_1 = lamda_0 -  aaa./bbb ;
    lamda_1 = abs(real(lamda_1));
    if max(abs(lamda_1-lamda_0))<thr*lamda_0
        break
    end
    lamda_0 = lamda_1;
end

% find the q(k) Integral upper and lower bounds, dichotomy used
for i = 1:staNum
    lamda_min_0 = lamda_1(i);    lamda_max_0 = lamda_1(i);      log_q_lamda_0 = fun_log_q_lamdai(W_lamda,Gamma,delay,lamda_min_0,IRF,i);
    lamda_min_1 = 0.95*lamda_min_0;   log_q_lamda_min = fun_log_q_lamdai(W_lamda,Gamma,delay,lamda_min_1,IRF,i);
    lamda_max_1 = 1.05*lamda_max_0;     log_q_lamda_max = fun_log_q_lamdai(W_lamda,Gamma,delay,lamda_max_1,IRF,i);
    
    while ( log_q_lamda_0 - log_q_lamda_min ) < 15          %  if q_k_0/q_k_min > e6
        lamda_min_0 = lamda_min_1;
        lamda_min_1 = 0.95*lamda_min_0;     log_q_lamda_min = fun_log_q_lamdai(W_lamda,Gamma,delay,lamda_min_1,IRF,i);
    end
    lamda_min = [lamda_min_1 lamda_min_0];     % dichotomy
    while abs( log_q_lamda_0 - log_q_lamda_min - 15) > 1
        lamda_min_1 = mean(lamda_min);
        log_q_lamda_min = fun_log_q_lamdai(W_lamda,Gamma,delay,lamda_min_1,IRF,i);
        lamda_min((log_q_lamda_0 - log_q_lamda_min < 15)+1) = lamda_min_1;
    end
    lamda_min = lamda_min_1;
    
    while ( log_q_lamda_0 - log_q_lamda_max ) < 15        %  if q_k_0/q_k_max > e6
        lamda_max_0 = lamda_max_1;
        lamda_max_1 = 1.05*lamda_max_0;      log_q_lamda_max = fun_log_q_lamdai(W_lamda,Gamma,delay,lamda_max_1,IRF,i);
    end
    lamda_max= [lamda_max_0 lamda_max_1];     % dichotomy
    while abs( log_q_lamda_0 - log_q_lamda_max - 15) > 1
        lamda_max_1 = mean(lamda_max);
        log_q_lamda_max = fun_log_q_lamdai(W_lamda,Gamma,delay,lamda_max_1,IRF,i);
        lamda_max((log_q_lamda_0 - log_q_lamda_max > 15)+1) = lamda_max_1;
    end
    lamda_max = lamda_max_1;
    
    x_lamda(i,:) = lamda_min+(0:sec_num)*((lamda_max-lamda_min)/sec_num);
end

% Passing parameters to GPU
gpu_log_q = zeros(staNum,size(x_lamda,2),'gpuArray');      % creat GPU var
gpuVar_lamda = gpuArray(x_lamda);
gpuVar_W = gpuArray(W_lamda);
gpuVar_Gamma = gpuArray(Gamma);
gpuVar_delay = gpuArray(delay);
gpuVar_x2 = gpuArray(x2);
gpuVar_IRF = gpuArray(IRF);
for i = 1:staNum     % get log_q_lamda
    gpu_log_q(i,:) = gpuVar_W(i,1).*log(gpuVar_lamda(i,:)/2) ...
          + gpuVar_W(i,2).*gpuVar_lamda(i,:)        ...
          + gpuVar_W(i,3).*gpuVar_lamda(i,:).^2     ...
          + gpuVar_Gamma(i,:)*log(erfc(    (gpuVar_IRF(1)-gpuVar_delay'+gpuVar_lamda(i,:)*gpuVar_IRF(2)^2)/(2^0.5*gpuVar_IRF(2))    ));
end

gpu_max_log_q = max(gpu_log_q,[],2);
gpu_q_nuNorm = exp(gpu_log_q-gpu_max_log_q);
gpu_AA =sum( ... 
             0.5*(gpu_q_nuNorm(:,1:end-1)+gpu_q_nuNorm(:,2:end))...
               .*(gpuVar_lamda(:,2:end)-gpuVar_lamda(:,1:end-1)) ...
             ,2);  % AA = [staNum, 1];
gpu_q =  gpu_q_nuNorm./gpu_AA;          q = gather(gpu_q);                           % normalized distribution
for i = 1:staNum
    gpu_Bmat(i,:) = exp(   0.5*((gpuVar_lamda(i,2:end)-gpuVar_lamda(i,1:end-1)).*gpu_q(i,1:end-1)) * fun_log_B_mat(gpuVar_x2,gpuVar_lamda(i,1:end-1)',gpuVar_IRF(1),gpuVar_IRF(2))...
                    + 0.5*((gpuVar_lamda(i,2:end)-gpuVar_lamda(i,1:end-1)).*gpu_q(i,2:end))   * fun_log_B_mat(gpuVar_x2,gpuVar_lamda(i,2:end)',gpuVar_IRF(1),gpuVar_IRF(2))...
                  );
end
% lamda = [lamda_1  lamda_2  ...];
lamda = (sum( ...
             0.5*(x_lamda(:,2:end)-x_lamda(:,1:end-1))...
               .*(x_lamda(:,1:end-1).*q(:,1:end-1)+x_lamda(:,2:end).*q(:,2:end))...
             ,2))';
gpu_log_Norm_q_temp = sum(gpu_max_log_q+log(gpu_AA));

B_mat = gather(gpu_Bmat);
log_Norm_q_temp = gather(gpu_log_Norm_q_temp);
q = [x_lamda; q];

end



