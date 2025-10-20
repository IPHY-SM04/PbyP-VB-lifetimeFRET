function [output_seq,Pfinal] = viterbi_H2MM_lifetime(A_lase,B_lase,pi_matrix)
% viterbi for H2MM, 
%   input A_lase,     B_lase,       Pi_matrix
%   (sta_num^2×N-1)  (sta_num×N)   （1×sta_num）
N=size(B_lase,2);    % number of photons
sta_num=size(B_lase,1);
log_B_lase=log(B_lase);
log_delte=zeros(sta_num,N);
Psi=log_delte;
output_seq=zeros(1,N);

%给Delte第一列赋值
for i=1:size(log_delte,1)
    log_delte(i,1)=log(pi_matrix(i))+log_B_lase(i,1);
end
%给Psi第一个赋值
Psi(:,1)=0;

%递推得到delte,Psi
for t=2:N     %对trace总时间序列循环 
    for i=1:sta_num   %对态循环
        A_matrix=A_lase(:,:,t-1);
        Max=max(log_delte(:,t-1)+log(A_matrix(:,i)));
        [row,~]=find((log_delte(:,t-1)+log(A_matrix(:,i)))==Max);
        log_delte(i,t)=Max+log_B_lase(i,t);
        Psi(i,t)=row;
    end
end
%最优路径回溯
Pfinal=max(log_delte(:,end));
[row,~]=find(log_delte(:,end)==Pfinal);
output_seq(end)=row;

for t=N-1:-1:1
    output_seq(t)=Psi(output_seq(t+1),t+1);
end

end
