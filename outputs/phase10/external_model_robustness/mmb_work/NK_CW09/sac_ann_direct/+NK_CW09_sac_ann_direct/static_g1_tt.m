function [T_order, T] = static_g1_tt(y, x, params, T_order, T)
if T_order >= 1
    return
end
[T_order, T] = NK_CW09_sac_ann_direct.static_resid_tt(y, x, params, T_order, T);
T_order = 1;
if size(T, 1) < 36
    T = [T; NaN(36 - size(T, 1), 1)];
end
T(26) = 1/y(5);
T(27) = getPowerDeriv(y(3),(-params(45)),1);
T(28) = 1/params(47)*getPowerDeriv(y(3)/params(47),T(19),1);
T(29) = getPowerDeriv(params(38)*T(20)+(1-params(38))*T(21),params(42),1);
T(30) = getPowerDeriv(y(4),(-params(46)),1);
T(31) = 1/params(48)*getPowerDeriv(y(4)/params(48),T(19),1);
T(32) = getPowerDeriv(y(5),(1+params(41))*params(43),1);
T(33) = getPowerDeriv(y(5),params(43)-1,1);
T(34) = getPowerDeriv(y(7)/y(21),1+params(41),1);
T(35) = getPowerDeriv(y(9)/y(8),(params(43)-1)/(1+params(41)*params(43)),1);
T(36) = getPowerDeriv(y(17),(-params(42)),1);
end
