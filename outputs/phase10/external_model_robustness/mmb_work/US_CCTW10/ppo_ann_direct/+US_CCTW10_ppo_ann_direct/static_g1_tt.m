function [T_order, T] = static_g1_tt(y, x, params, T_order, T)
if T_order >= 1
    return
end
[T_order, T] = US_CCTW10_ppo_ann_direct.static_resid_tt(y, x, params, T_order, T);
T_order = 1;
if size(T, 1) < 14
    T = [T; NaN(14 - size(T, 1), 1)];
end
T(13) = 1-(T(6)/(1+T(6))+1/(1+T(6)));
T(14) = 1/(1-T(6))-T(6)/(1-T(6));
end
