function [T_order, T] = dynamic_g1_tt(y, x, params, steady_state, T_order, T)
if T_order >= 1
    return
end
[T_order, T] = SW_US_fiscal.dynamic_resid_tt(y, x, params, steady_state, T_order, T);
T_order = 1;
if size(T, 1) < 17
    T = [T; NaN(17 - size(T, 1), 1)];
end
T(17) = 0.5*(T(1))^(-0.5);
end
