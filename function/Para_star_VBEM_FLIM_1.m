function [k,zeta,Pi,log_Norm_qk,q_k] = Para_star_VBEM_FLIM_1(trace_PbyP,W_k,V_k,W_zeta,W_Pi)
% E step for VBEM
% input: W_k,V_k,W_zeta,W_Pi,W_B
% output: Pi*, k*, zeta*,B*

sec_num = 500;   % 

dT = trace_PbyP(3,2:end);
staNum = length(W_k);
k = zeros(1,staNum);
dt = 1e-6; log_dt = log(dt);
k_0 = sum(V_k,2)./((W_k+dt)'+V_k*dT');        % Seeking Extreme Values of k
fun_log_q_ki = @(W_k_i,V_k_i,dt,dT,k)-(W_k_i+dt)*k + V_k_i*log(1-exp(-dT*k));
x_k = zeros(staNum,sec_num+1);       % q_k{staNum} = [x_k, q(k)];
% get the q(k) Integral upper and lower bounds

% Find Extreme Values of ln(q(k)), Newton's method  x_n+1 = x_n - f'(x_n)/f''(x_n)
thr = 1e-6;

for iter = 1:10000
    % Newton's method  x_n+1 = x_n - f'(x_n)/f''(x_n)
    aaa = -(W_k + dt)' + sum( (V_k.*dT)./(exp(k_0.*dT)-1), 2);
    bbb = - sum( (V_k.*dT.*dT.*exp(k_0.*dT))./(exp(k_0.*dT)-1).^2, 2);
    k_1 = k_0 -  aaa./bbb ;
    k_1 = abs(real(k_1));
    if sum(abs(k_1-k_0) > thr*k_0) == 0
        break
    end
    k_0 = k_1;
end
k_0 = k_1;
log_q_k_0 = -(W_k'+dt).*k_0 + sum( V_k.*log(1-exp(-dT.*k_0)),2);

% Find Zero point coordinates of ln(q(k))-ln(q(k_0))+20, Newton's method  x_n+1 = x_n - f(x_n)/f''(x_n)
k_min_0 = 0.01*k_0; 
k_max_0 = 100*k_0; 
for iter = 1:10000
    % k_min
    % Newton's method  x_n+1 = x_n - f(x_n)/f'(x_n)
    aaa = -(W_k'+dt).*k_min_0 + sum( V_k.*log(1-exp(-dT.*k_min_0)),2) - log_q_k_0 + 20;
    bbb =  -(W_k+dt)' + sum( (V_k.*dT)./(exp(k_min_0.*dT)-1), 2);
    k_min_1 = k_min_0 -  aaa./bbb ;
    k_min_1 = abs(real(k_min_1));
    if sum(abs(k_min_1-k_min_0) > thr*k_min_0) == 0
        break
    end
    k_min_0 = k_min_1;
end
for iter = 1:10000
    % k_max
    % Newton's method  x_n+1 = x_n - f(x_n)/f'(x_n)
    aaa = -(W_k'+dt).*k_max_0 + sum( V_k.*log(1-exp(-dT.*k_max_0)),2) - log_q_k_0 + 20;
    bbb =  -(W_k+dt)' + sum( (V_k.*dT)./(exp(k_max_0.*dT)-1), 2);
    k_max_1 = k_max_0 -  aaa./bbb ;
    k_max_1 = abs(real(k_max_1));
    if sum(abs(k_max_1-k_max_0) > thr*k_max_0) == 0
        break
    end
    k_max_0 = k_max_1;
end
for i = 1:staNum
    x_k(i,:) = k_min_0(i)+(0:sec_num)*((k_max_0(i)-k_min_0(i))/sec_num);
end


log_Norm_qk=zeros(1,staNum);
q_k = zeros(staNum,size(x_k,2));       % [x; q_k1; ...; q_ki; ...]
gpuVar = zeros(staNum,size(x_k,2),'gpuArray');      % creat GPU var
gpuVar_V_k = gpuArray(V_k);
gpuVar_W_k = gpuArray(W_k');
gpuVar_dT = gpuArray(dT');
gpuVar_x_k = gpuArray(x_k);
gpuVar_dt = gpuArray(dt);
for i = 1:staNum
    gpuVar(i,:) = -(gpuVar_W_k(i)+gpuVar_dt)*gpuVar_x_k(i,:) + gpuVar_V_k(i,:)*log(1-exp(-gpuVar_dT*gpuVar_x_k(i,:)));     
end
log_q_k = gather(gpuVar)+log_dt;
max_log_q_k = max(log_q_k,[],2);
q_k_unNorm=exp(log_q_k-max_log_q_k); 
AA = sum( 0.5*(q_k_unNorm(:,1:end-1)+q_k_unNorm(:,2:end)).*(x_k(:,2:end)-x_k(:,1:end-1)),2);
log_Norm_qk = max_log_q_k + log(AA);
q_k = q_k_unNorm./(AA);   % normalized distribution
k = exp(  sum(  0.5*(log(x_k(:,1:end-1)).*q_k(:,1:end-1)+log(x_k(:,2:end)).*q_k(:,2:end))...
        .*(x_k(:,2:end)-x_k(:,1:end-1)), 2)  );                       % Trapezoidal numerical integration
k=k';
q_k = [x_k;q_k];
log_Norm_qk=sum(log_Norm_qk);

% zeta*
W_zeta_0 = sum(W_zeta-diag(diag(W_zeta)),2);
zeta = exp(psi(W_zeta)-psi(W_zeta_0));
zeta = zeta - diag(diag(zeta)) + diag(ones(1,staNum));
% Pi*
W_Pi_0 = sum(W_Pi);
Pi = exp(psi(W_Pi)-psi(W_Pi_0));


end




