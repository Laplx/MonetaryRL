function [T_order, T] = dynamic_g1_tt(y, x, params, steady_state, T_order, T)
if T_order >= 1
    return
end
[T_order, T] = NK_CW09_empirical_taylor_rule.dynamic_resid_tt(y, x, params, steady_state, T_order, T);
T_order = 1;
if size(T, 1) < 36
    T = [T; NaN(36 - size(T, 1), 1)];
end
T(27) = getPowerDeriv(y(82),(-params(45)),1);
T(28) = 1/params(47)*getPowerDeriv(y(82)/params(47),T(20),1);
T(29) = getPowerDeriv(params(38)*T(21)+(1-params(38))*T(22),params(42),1);
T(30) = 1/y(163);
T(31) = getPowerDeriv(y(83),(-params(46)),1);
T(32) = 1/params(48)*getPowerDeriv(y(83)/params(48),T(20),1);
T(33) = (-(params(44)*getPowerDeriv(y(84),params(43)-1,1)))/(1-params(44));
T(34) = getPowerDeriv(y(86)/y(100),1+params(41),1);
T(35) = getPowerDeriv(y(88)/y(87),(params(43)-1)/(1+params(41)*params(43)),1);
T(36) = getPowerDeriv(y(96),(-params(42)),1);
end
