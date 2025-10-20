function [W] = P_W_lamda_VBEM(Gamma,u_lamda,trace_PbyP,IRF)
W_0 = sum(Gamma,2);
W_1 = W_0+u_lamda(:,1)-1;
W_2 = IRF(1)*W_0-u_lamda(:,2)-Gamma*trace_PbyP(1,:)';
W_3 = 0.5*IRF(2)^2*W_0;
W = [W_1 W_2 W_3];
end

