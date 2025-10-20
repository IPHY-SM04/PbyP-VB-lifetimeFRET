
%蒙特卡洛模拟A-B-C三个态之间的变换，得到大量状态变化trace（100），来进行后续的寿命模拟运算

clear
% clc



hidSta_num = 2;
time_Trace = 50;         % ms
Trace_num  = 10;
dt = 10^(-6);               % dt=1 us
dwell_thr = 0;              % ms
length_Trace = time_Trace*1000;
Trace = zeros(Trace_num,length_Trace);                           % 每步代表，即1000步代表1ms
connect1 = zeros(Trace_num,length_Trace);

K_set = [300 300];
Pi_set = (1./K_set(1:hidSta_num));
% Pi_set = [0.2 0.1 0.8];
Pi_set = Pi_set./sum(Pi_set);
kappa = zeros(hidSta_num,hidSta_num)+1/(hidSta_num-1);
% kappa = [0 1 0 ;
%          1/6 0 5/6;
%          0 1 0];
% kappa = [0 0.5 0.5;
%          2000/3000 0 1000/3000;
%          2000/3000 1000/3000 0];
% kappa = [0 1 0 0;
%          0.5 0 0.5 0;
%          0 0.5 0 0.5;
%          0 0 1 0];
% kappa = [0 1 0 0 0;
%          0.5 0 0.5 0 0;
%          0 0.5 0 0.5 0;
%          0 0 0.5 0 0.5;
%          0 0 0 1 0];
kappa = kappa-diag(diag(kappa));
kappa = kappa./sum(kappa,2);
trans_matrix=zeros(hidSta_num,hidSta_num);                        % 创建状态转移矩阵
for i=1:hidSta_num
    for j=1:hidSta_num
        if i==j
            trans_matrix(i,j)=exp(-K_set(i)*dt);
        else
            trans_matrix(i,j)=kappa(i,j)*(1-exp(-K_set(i)*dt));
        end
    end
end

%%
% 马尔可夫过程
% 给初始态(前两个)赋值
Prob_Pi=zeros(1,hidSta_num);
Prob_Pi(end)=1;
for i=1:hidSta_num-1
    Prob_Pi(i)=sum(Pi_set(:,1:i));
end
P = rand(Trace_num,1);
for i = 1:Trace_num
    for j = 1:hidSta_num
        if P(i) <= Prob_Pi(j)
            Trace(i,1)=j;
            break
        end
    end
end

% 计算Prob矩阵，第n列为前n个概率之和，用来后面判断
Prob=zeros(hidSta_num,hidSta_num);
Prob(:,hidSta_num)=1;
for i=1:hidSta_num-1
    Prob(:,i)=sum(trans_matrix(:,1:i),2);
end
trans_P = 0;
for n = 1:Trace_num
    for i=1:length_Trace-1
        P = rand;
        sta_n = Trace(n,i);    %获取当前隐状态
        for j=1:hidSta_num
            if P<=Prob(sta_n,j)
                Trace(n,i+1)=j;
                break
            end
        end
    end
end
connect1(:,end) = connect1(:,end)+1;
Trace = reshape(Trace',1,[]);
connect1 = reshape(connect1',1,[]);
%%
%show Trace

Trace_x = 1:1:length(Trace);
Trace_1 = Trace(Trace_x);
figure
plot(Trace_x*1e-3,Trace_1,'LineWidth',2)
ylim([0.5,hidSta_num+0.5])
hold on
plot((1:1:length(Trace))*1e-3,connect1*10)


% K_real and K_app
trans_mat_app = zeros(hidSta_num,hidSta_num);
trans_mat_real = zeros(hidSta_num,hidSta_num);
for i=1:length(Trace)-1
    trans_mat_app(Trace(i),Trace(i+1))=trans_mat_app(Trace(i),Trace(i+1))+1;
    if connect1(i) == 0
        trans_mat_real(Trace(i),Trace(i+1))=trans_mat_real(Trace(i),Trace(i+1))+1;
    end
end
trans_mat_app=trans_mat_app./sum(trans_mat_app,2);
trans_mat_real=trans_mat_real./sum(trans_mat_real,2);
K_app = -log(diag(trans_mat_app))/dt;
K_real = -log(diag(trans_mat_real))/dt;



%%
% clearvars -except hidSta_num trans_matrix Trace dt trans_mat_real K_real K_app dt connect1;

%close all


% aaa = mod(aaa,24.99);
% aaa1 = aaa(aaa ~= 0);   aaa1 = floor(aaa1/resolution)*resolution;   
% aaa2 = find(aaa ~= 0)*1e-6;
% aaa3 = aaa2(2:end)-aaa2(1:end-1);
% trace_PbyP = [aaa1'; aaa2';[0 aaa3']];
% trace_PbyP(4,end) = 1;
% max(trace_PbyP(1,:))
