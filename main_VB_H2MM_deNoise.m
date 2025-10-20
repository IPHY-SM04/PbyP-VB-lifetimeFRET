%% VBEM + photon by photon HMM for lifetime data
% estimate the most likely state number and dynamic parameters
%
%% initialization: Parameter Settings

% Instrument parameter
IRF = [2.5 0.2];       % The instrument response function of TCSPC including [Peak position, standard deviation]
maxTau = 25;           % The maximum micro time of photons.  time unit: ns.
resolution = 0.016;    % Time resolution of TCSPC.           time unit: ns.

% EM algorithm iteration setting
staNum_min = 2;         % Minimum number of states
staNum_max = 3;         % Maximum number of states
maxIter = 3000;         % Iteration higher limit
minIter = 2;            % Iteration lower limit
if_break_thre = 1e-5;   % Iterative termination threshold


%% data import
% photon data should be *.txt file
% data format: Three columns of data are respectively: micro time(ns), macro time(s), if the photon end of trace(0 or 1)

[file_name,path_name] = uigetfile('.txt','Select a text file');
fprintf(['当前文件为：',num2str(file_name),'\n']);
[fid,message] = fopen([path_name file_name],'rt');
rawdata = textscan(fid,'%f %f %f','delimiter',',');
trace_PbyP_deNoise = [rawdata{1} rawdata{2} rawdata{3}]';
trace_PbyP_deNoise(4,2:end) = trace_PbyP_deNoise(2,2:end) - trace_PbyP_deNoise(2,1:end-1);
trace_PbyP_deNoise([3 4],:) = trace_PbyP_deNoise([4 3],:);
% trace_PbyP_deNoise = [micro time(ns);
%               macro time(s);
%               Interval time between adjacent photons(s);
%               if the photon end of trace(0 or 1)];

%%
N = size(trace_PbyP_deNoise,2);                   % total photon counts
M = sum(trace_PbyP_deNoise(4,:));                 % total counts of burst/trace connected
end_pos = find(trace_PbyP_deNoise(4,:) == 1);
start_pos = [1 end_pos(1:end-1)+1];

delay_t = resolution:resolution:maxTau;
fun_B_mat = @(x,k,a1,a2)k/2.*exp(k.*(a1-x)+0.5.*k.^2.*a2.^2).*erfc((a1-x+k.*a2.^2)./(2^0.5*a2));  % Photon micro time decay function

%% Variables for display && debugging
staNum_lase  = 1:staNum_max;
iter_num = zeros(1,staNum_max);                  % final iterations of different staNum situation

log_P_staNum = zeros(maxIter,staNum_max);        % Logarithmic marginal likelihood  varies with the number of iteration under different models
log_P_staNum_final = zeros(1,staNum_max)-Inf;    % ln(P(t|M))：Logarithmic marginal likelihood  at iteration termination under different models

KL_Pi_final = zeros(1,staNum_max)-Inf;
KL_k_final = zeros(1,staNum_max)-Inf;
KL_zeta_final = zeros(1,staNum_max)-Inf;
KL_lamda_final = zeros(1,staNum_max)-Inf;

% parameters varies with the number of iteration under different models
K_fit = cell(maxIter,length(staNum_lase));
zeta_fit = cell(maxIter,length(staNum_lase));
B_mat_fit = cell(maxIter,length(staNum_lase));
lamda_fit = cell(maxIter,length(staNum_lase));
seq_fit = cell(1,length(staNum_lase));

% q(φ) varies with the number of iteration under different models
q_k_fit = cell(maxIter,length(staNum_lase));
W_Pi_fit = cell(maxIter,length(staNum_lase));
W_zeta_fit = cell(maxIter,length(staNum_lase));
W_lamda_fit = cell(maxIter,length(staNum_lase));
q_lamda_fit = cell(maxIter,length(staNum_lase));

%% Evaluate models with different states numbers in sequence
% staNum is 1,the only variable is lamda,that is lifetime

u_lamda = ones(1,2);
Gamma = ones(1,size(trace_PbyP_deNoise,2));
W_lamda =P_W_lamda_VBEM(Gamma,u_lamda,trace_PbyP_deNoise,IRF);  % used for q(lamda)
[B_mat,lamda_1,log_Norm_qlamda_temp,q_lamda] = B_mat_VBEM_FLIM(trace_PbyP_deNoise,W_lamda,Gamma,IRF,maxTau,resolution,1);
log_Norm_qlamda = log_Norm_qlamda_temp+u_lamda(1)*log(u_lamda(2))-log(gamma(u_lamda(1)))+sum(u_lamda(:,1)-1)*log(2);
log_P_staNum_final(1) = log_Norm_qlamda;
q_lamda_fit{1,1} = q_lamda;
lamda_fit{1,1} = lamda_1;


% staNum_min ≤ staNum ≤ staNum_max
for staNum = staNum_lase(staNum_min:staNum_max)
    %% set hyperparameters for the probability: P(Pi),P(k),P(zeta),P(lamda)
    % P(Pi),P(zeta)is Dirichlet distribution, P(k)=dt*exp(-dt*k),
    % P(lamda) is Gamma distribution,define by Gamma(a,b)
    u_Pi = ones(1,staNum);  u_zeta = ones(staNum,staNum);  u_lamda = ones(staNum,2); 
    
    % q(Z) is defined by  Pi*, k*, zeta*(ζ),lamda*(λ), which are initialized below
    k_0 = zeros(1,staNum)+100;
    zeta_0 = zeros(staNum,staNum);
    for i = 1:staNum
        zeta_0(i,:) = zeros(1,staNum)+1/(staNum-1);
    end
    Pi_0    = zeros(1,staNum)+1/staNum; 
    
    %%%%%% lamda_0 may have a significant impact in certain situations and
    %%%%%% can be selected by oneself 
    lamda_0 = 1./((1:staNum)*1.5);

    B_mat       =  fun_B_mat(delay_t,lamda_0',IRF(1),IRF(2));     % p(t|τ): Micro time decay curves with different lifetimes
    A_lase      =  zeros(staNum,staNum,N-1);                      % Transition probability matrix of adjacent photons;
    A_lase_temp =  permute(A_lase,[3,2,1]);
    %% EM iteration
    fig_mntr = figure;
    plot_mntr = plot([1,2],[0 0]);
    
    for iter = 1:maxIter
        %%%%%%%% M step %%%%%%%%
        disp([staNum iter])
        % Pi*, k*, zeta*,B* → W_k,V_k,W_Pi,W_B
        for k = 1:staNum
            temp = exp(-k_0(k)*trace_PbyP_deNoise(3,2:end)');
            A_lase_temp(:,:,k) = (1-temp).*zeta_0(k,:);
            A_lase_temp(:,k,k) = temp;
        end
        A_lase = permute(A_lase_temp,[3,2,1]);
        for m = 1:M-1
            A_lase(:,:,end_pos(m)) = Pi_0.*ones(staNum,1);
        end
        
        % B_lase(i,j) ≡ p(t_j |z_j = i), 
        B_lase = B_mat(:,floor(trace_PbyP_deNoise(1,:)/resolution)+1);
        [Alpha,Norm_Alpha] = forward_algo_VBEM(A_lase,B_lase,Pi_0);            % forward algorithm
        Beta      = backward_algo_VBEM(A_lase,B_lase);                         % backward algorithm
        Gamma     = P_Gamma_VBEM(Alpha,Beta);                                  % equation S27 in SI, ω_i^n=∑_z q(z)δ(z_n=i)
        Xi        = P_Xi_VBEM(trace_PbyP_deNoise,A_lase,B_lase,Alpha,Beta);    % equation S26 in SI, ω_ij^n=∑_z q(z)δ(z_n=i,z_(n+1)=j)
        
        
        [W_k,V_k] = P_WV_k_VBEM_connect(trace_PbyP_deNoise,Xi);                % used for q(k), see equations S23 & S24
        W_lamda   = P_W_lamda_VBEM(Gamma,u_lamda,trace_PbyP_deNoise,IRF);      % used for q(λ), 
        W_Pi    = (sum(Gamma(:,start_pos),2))'+u_Pi;                           % used for q(π), see equation S22
        W_zeta = sum(Xi,3)-sum(Xi(:,:,end_pos(1:end-1)),3)+u_zeta;             % used for q(ζ), see equation S25
        
        %%%%%%%% E step %%%%%%%%%
        % W_k,V_k,W_Pi,W_zeta → Pi*, k*, zeta*,B*
        [k_1,zeta_1,Pi_1,log_Norm_qk,q_k] = Para_star_VBEM_FLIM_1(trace_PbyP_deNoise,W_k,V_k,W_zeta,W_Pi);
        % W_lambda → lambda*
        [B_mat,lamda_1,log_Norm_qlamda_temp,q_lamda] = B_mat_VBEM_FLIM_1(trace_PbyP_deNoise,W_lamda,Gamma,IRF,maxTau,resolution,lamda_0);
        
        % calculate ln(P(t|M))
        log_Norm_qz = sum(log(Norm_Alpha))+log(sum(Alpha(:,end)));
        [log_P_staNum(iter,staNum),KL_Pi,KL_k,KL_zeta,KL_lamda] = log_P_XofstaNum_FLIM(log_Norm_qz,q_k,q_lamda,W_zeta,W_Pi,u_Pi,u_zeta);
        
        k_0 = k_1;
        zeta_0 = zeta_1;
        Pi_0 = Pi_1;
        lamda_0 = lamda_1;
        
        q_k_fit{iter,staNum} = q_k;
        W_Pi_fit{iter,staNum} = W_Pi;
        W_zeta_fit{iter,staNum} = W_zeta;
        q_lamda_fit{iter,staNum} = q_lamda;
        W_lamda_fit{iter,staNum} = W_lamda;
        
        K_fit{iter,staNum} = k_0;
        zeta_fit{iter,staNum} = zeta_0;
        B_mat_fit{iter,staNum} = B_mat;
        lamda_fit{iter,staNum} = lamda_0;
        
        % Determine whether to terminate the iteration
        set(plot_mntr,'XData',(1:iter),'YData',real(log_P_staNum(1:iter,staNum)));
        drawnow
        if iter >= minIter
            if abs((log_P_staNum(iter,staNum)-log_P_staNum(iter-1,staNum))/log_P_staNum(iter,staNum)) < if_break_thre
                break
            end
        end
        
    end
    log_P_staNum_final(staNum) = real(log_P_staNum(iter,staNum));
    iter_num(1,staNum) = iter;
    [seq_fit{staNum},~] = viterbi_H2MM_lifetime(A_lase,B_lase,Pi_0);  % Viterbi algorithm estimates optimal 
    
end

%%

% best fitted state number, parameters, hidden state
staNum_best = find(log_P_staNum_final(staNum_min:staNum_max)==max(log_P_staNum_final(staNum_min:staNum_max)))+staNum_min-1;
staNum_best = staNum_lase(staNum_best);
k_best = K_fit{iter_num(staNum_best),staNum_best};
zeta_best = zeta_fit{iter_num(staNum_best),staNum_best};
tau_best = 1./lamda_fit{iter_num(staNum_best),staNum_best};
seq_best = seq_fit{staNum_best};

% result present
figure
plot(staNum_lase(1:staNum_max),log_P_staNum_final(1:staNum_max))
title('ln(P(t|M))'); xlabel('state number M'); ylabel('ln(P(t|M))');

figure
plot(trace_PbyP_deNoise(2,:),seq_best,'-*')
hold on
plot(trace_PbyP_deNoise(2,:),trace_PbyP_deNoise(4,:)*10)
ylim([0 staNum_best+3])
title('state sequence'); xlabel('Macro time (s)'); ylabel('state');

% q_tau and plot(q_tau)
q_tau = cell(1,length(staNum_lase));
for staNum = staNum_min:staNum_max
    temp = q_lamda_fit{iter_num(staNum),staNum};
    temp(staNum+1:end,:) = temp(staNum+1:end,:).*temp(1:staNum,:).^2;
    temp(1:staNum,:) = 1./temp(1:staNum,:);
    q_tau{staNum} = temp(:,end:-1:1);
end
for staNum = staNum_min:staNum_max
    figure
    for fig = 1:staNum
        plot(q_tau{staNum}(fig,:),q_tau{staNum}(fig+staNum,:))
        hold on
    end
    title(['q(tau|staNum=' num2str(staNum) ')'])
    xlabel('lifetime tau (ns)'); ylabel('q(tau)');
end



