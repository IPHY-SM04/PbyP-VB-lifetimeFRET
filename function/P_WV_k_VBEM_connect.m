function [W_k,V_k] = P_WV_k_VBEM_connect(trace_PbyP,Xi)

% W is 1 * state num,
% V is  state num * N-1(N: number of photons )
% M = sum(trace_PbyP(4,:));    % counts of burst connected

end_pos = find(trace_PbyP(4,:) == 1);
% start_pos = [1 end_pos(1:end-1)+1];
sta_num = size(Xi,1);
Xi_temp = permute(Xi,[3 1 2]);
W_k     = zeros(1,sta_num);
% V_k=zeros(sta_num,N);

for i=1:sta_num
    W_k(i)=  trace_PbyP(3,2:end)*Xi_temp(:,i,i) ...
            -trace_PbyP(3,end_pos(1:end-1)+1)*Xi_temp(end_pos(1:end-1),i,i);
    Xi_temp(:,i,i) = 0;
end
Xi_dediag =  permute(Xi_temp,[2 3 1]);
V_k_temp  =  sum(Xi_dediag,2);
V_k       =  squeeze(V_k_temp);
V_k(:,end_pos(1:end-1)) = 0;
end

