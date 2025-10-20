function [gamma] = P_Gamma_VBEM(forward_prob,backward_prob)
% input forward probability, backward probability,T(nummber of photon)
% forward probability: sta_num × T matrix
% backward probability: sta_num × T matrix
% output  probability of being in state i at time t

gamma_temp = forward_prob.*backward_prob;        % not normalized gamma
gamma = gamma_temp./sum(gamma_temp);
end


