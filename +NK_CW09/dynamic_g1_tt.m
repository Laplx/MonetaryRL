function [T_order, T] = dynamic_g1_tt(y, x, params, steady_state, T_order, T)
if T_order >= 1
    return
end
[T_order, T] = NK_CW09.dynamic_resid_tt(y, x, params, steady_state, T_order, T);
T_order = 1;
if size(T, 1) < 57
    T = [T; NaN(57 - size(T, 1), 1)];
end
T(29) = 1/y(84);
T(30) = getPowerDeriv(y(82),(-params(45)),1);
T(31) = 1/params(47)*getPowerDeriv(y(82)/params(47),T(22),1);
T(32) = getPowerDeriv(params(38)*T(23)+(1-params(38))*T(24),params(42),1);
T(33) = 1/y(163);
T(34) = getPowerDeriv(y(83),(-params(46)),1);
T(35) = 1/params(48)*getPowerDeriv(y(83)/params(48),T(22),1);
T(36) = (-(1+y(1)))/(y(84)*y(84));
T(37) = getPowerDeriv(y(84),(1+params(41))*params(43),1);
T(38) = (-(params(44)*getPowerDeriv(y(84),params(43)-1,1)))/(1-params(44));
T(39) = getPowerDeriv(T(17),T(18),1);
T(40) = (params(37)+(1-params(37))*params(38))*(-y(161))/(y(163)*y(163))+(1-params(37))*(1-params(38))*(-y(162))/(y(163)*y(163));
T(41) = (1-params(37))*params(38)*(-y(161))/(y(163)*y(163))+(params(37)+(1-params(37))*(1-params(38)))*(-y(162))/(y(163)*y(163));
T(42) = params(35)*params(44)*getPowerDeriv(y(163),(1+params(41))*params(43),1);
T(43) = params(35)*params(44)*getPowerDeriv(y(163),params(43)-1,1);
T(44) = 1/y(100);
T(45) = getPowerDeriv(y(86)/y(100),1+params(41),1);
T(46) = T(44)*T(45);
T(47) = (-y(88))/(y(87)*y(87));
T(48) = getPowerDeriv(y(88)/y(87),T(19),1);
T(49) = getPowerDeriv(y(90),params(50),1);
T(50) = getPowerDeriv(y(90),params(51),1);
T(51) = getPowerDeriv(y(90),params(50)-1,1);
T(52) = getPowerDeriv(y(96),(-params(42)),1);
T(53) = T(45)*(-y(86))/(y(100)*y(100));
T(54) = getPowerDeriv(y(104),(-1),1);
T(55) = y(103)*params(40)*(1+params(41))*params(49)*y(97)*T(54);
T(56) = T(5)*T(55);
T(57) = 1/params(49)*getPowerDeriv(y(104)/params(49),(-(1+params(42)))/params(42),1);
end
