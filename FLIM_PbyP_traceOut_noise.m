%%根据模拟的状态变化曲线，得到延迟τ和宏观时间T的曲线

% 设定各项参数
resolution=0.025;                 % 仪器分辨率 resolution=0.016 ns
IRF = [2.5 0.2];                  % IRF = [expectation standard deviation]


trace = zeros(size(Trace,1),size(Trace,2))+Inf; 
if_noise = zeros(size(Trace,1),size(Trace,2));
tau_set = [1.5 3];                                            % 1，2，3（或 4 5）三（或四，五）个状态的荧光寿命
eOfExc = tau_set(1:hidSta_num)/max(tau_set(1:hidSta_num));      % tau/max(max(tau));    
eOfExc = eOfExc.^0.5;
% eOfExc = ones(1,hidSta_num);
eOfRec = 0.1;                                                 % 光子接收率
bg_noise_rate_1 = 0.002;                                      % 暗噪声/ms，均匀分布
bg_noise_rate_2 = 0.00;                                      % 杂志噪声，指数衰减，与IRF无关
max_tau= 24.99;                                               % 最大能探测到的delay time

%%
% 循环开始，对于每次激光激发，有概率接收到光子信号
P = rand(4,length(trace));  % 三行概率分别 暗噪音 接受光概率 delaytime的概率，指数噪音 
noise = 0;
Pho_fluor = 0;
Pho_noise = 0;
SNR = [];
phoCount_ms = [];


for i=1:size(trace,2)
    signal = Inf;
    noise_1 = Inf;
    noise_2 = Inf;
    if P(2,i) < eOfExc(1,Trace(i))*eOfRec          % 判断是否激发并收到了光子
        State = Trace(i);
        signal = - normrnd(tau_set(State),tau_set(State)*0.0001,1,1)  *  log(1-P(3,i));
        signal = - tau_set(State)  *  log(1-P(3,i));
        % trace(i) = expDistr(normrnd(tau(State),tau(State)*0.1,1,1));
        % 将确定delay time的光子变为IRF形状分布的随即光子
        signal = normrnd(IRF(1)+signal,IRF(2),1,1);        
    end
    
    if P(1,i) < bg_noise_rate_1         % 判断是否有 暗噪音
            noise_1 = max_tau*rand(1);
    end
    
    if P(4,i) < bg_noise_rate_2         % 判断是否有 指数噪音
         noise_2 = - 5  *  log(1-rand(1));
    end
    
    trace(i) = min([signal noise_1 noise_2]);
    if trace(i) < Inf
        Pho_fluor = Pho_fluor + 1;
        switch find([signal noise_1 noise_2] == trace(i)) 
            case 1              
            otherwise
                Pho_noise = Pho_noise + 1; 
                if_noise(i) = 1;
        end
    end
end
%%

% 实际实验数据需满足时间分辨率
trace = floor(trace/resolution)*resolution;
trace(trace>max_tau) = Inf;
trace(trace<0) = Inf;
% trace(trace<(IRF(1)-3*IRF(2))) = Inf;
% 将trace中小于0的点去掉
trace_no_zero=trace(trace<Inf);   
% 信噪比 signal-to-noise ratio
SNR = Pho_fluor/Pho_noise;
% photons per ms
phoCount_ms = length(trace_no_zero)/length(trace)*1000;
signal_P = find(if_noise == 0);
noise_P = find(if_noise == 1);
trace_deNoise = trace(signal_P);
trace_noise = trace(noise_P);

edge=0:resolution:40;
decay_t=edge(1:end-1);
decay_real = histcounts(trace_no_zero,edge);
decay_real_deNoise = histcounts(trace_deNoise,edge);
decay_real_noise = histcounts(trace_noise,edge);

decay4origin=[decay_t',decay_real'];
figure
semilogy(decay_t,decay_real);
hold on
semilogy(decay_t,decay_real_deNoise);
hold on
semilogy(decay_t,decay_real_noise);
% ylim([-0.5,inf])


tau_1 = 3.99244;
IRF=decay_real-[0 decay_real(1:end-1)]*exp((decay_t(1)-decay_t(2))/tau_1);
IRF=IRF/sum(IRF);  % normalization
IRF(end)=0; IRF=IRF/sum(IRF);
Distr_IRF=@(x)exp(-(x-2.5).^2/0.032);
IRF_0=Distr_IRF(decay_t);
IRF_0=IRF_0/sum(IRF_0);
figure
plot(decay_t,IRF)
hold on
plot(decay_t,IRF_0)



% clearvars -except    hidSta_num trans_matrix Trace dt trans_mat_real K_real dt...
%                      resolution trace if_noise tau eOfRec eOfExc s2n_ratio;
pho_T = find(trace < Inf);
trace_no_zero = trace(trace < Inf);
pho_dT = [0 pho_T(2:end)-pho_T(1:end-1)]*dt;

connect_pos = find(connect1 == 1);
connect_1 = zeros(1,length(pho_T));
connect_1(end) = 1;
for i = 1:length(connect_pos)-1 
    temp = pho_T(pho_T<=connect_pos(i));
    connect_1(length(temp)) = 1;
%     pho_dT(length(temp)) = 0.01;
end

trace_PbyP = [trace_no_zero;pho_T*dt;pho_dT;connect_1];   % delay time, arrival time, interval time




figure
plot(trace_PbyP(2,:),trace_PbyP(1,:),'.')
hold on
plot(trace_PbyP(2,:),Trace(pho_T)*10)
hold on
plot(trace_PbyP(2,:),connect_1*max_tau)
ylim([0 max_tau*1.2])

%%
% VBEM_FLIM_connect_noise
% VBEM_FLIM_connect
% close all
% 
% figure
% plot(staNum_lase(1:staNum_max),log_P_staNum_final(1:staNum_max))
% hold on
% plot(staNum_lase(1:staNum_max),ICL_final(1:staNum_max),'-*')








