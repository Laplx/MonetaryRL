function [T_order, T] = dynamic_g2_tt(y, x, params, steady_state, T_order, T)
if T_order >= 2
    return
end
[T_order, T] = NK_CW09.dynamic_g1_tt(y, x, params, steady_state, T_order, T);
T_order = 2;
if size(T, 1) < 69
    T = [T; NaN(69 - size(T, 1), 1)];
end
T(58) = (-1)/(y(163)*y(163));
T(59) = getPowerDeriv(y(86)/y(100),1+params(41),2);
T(60) = T(45)*(-1)/(y(100)*y(100))+T(44)*(-y(86))/(y(100)*y(100))*T(59);
T(61) = getPowerDeriv(y(96),(-params(42)),2);
T(62) = (-y(86))/(y(100)*y(100))*(-y(86))/(y(100)*y(100))*T(59)+T(45)*(-((-y(86))*(y(100)+y(100))))/(y(100)*y(100)*y(100)*y(100));
T(63) = getPowerDeriv(y(82),(-params(45)),2);
T(64) = getPowerDeriv(y(83),(-params(46)),2);
T(65) = (-(params(44)*getPowerDeriv(y(84),params(43)-1,2)))/(1-params(44));
T(66) = getPowerDeriv(y(88)/y(87),T(19),2);
T(67) = 1/params(47)*1/params(47)*getPowerDeriv(y(82)/params(47),T(22),2);
T(68) = 1/params(48)*1/params(48)*getPowerDeriv(y(83)/params(48),T(22),2);
T(69) = getPowerDeriv(params(38)*T(23)+(1-params(38))*T(24),params(42),2);
end
