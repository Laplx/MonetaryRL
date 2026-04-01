function [T_order, T] = static_g1_tt(y, x, params, T_order, T)
if T_order >= 1
    return
end
[T_order, T] = US_SW07_US_SW07_baseline.static_resid_tt(y, x, params, T_order, T);
T_order = 1;
if size(T, 1) < 13
    T = [T; NaN(13 - size(T, 1), 1)];
end
T(12) = 1-(T(5)/(1+T(5))+1/(1+T(5)));
T(13) = 1/(1-T(5))-T(5)/(1-T(5));
end
