function [T_order, T] = dynamic_resid_tt(y, x, params, steady_state, T_order, T)
if T_order >= 0
    return
end
T_order = 0;
if size(T, 1) < 28
    T = [T; NaN(28 - size(T, 1), 1)];
end
T(1) = (params(37)+(1-params(37))*params(38))*y(161)/y(163)+(1-params(37))*(1-params(38))*y(162)/y(163);
T(2) = (1-params(37))*params(38)*y(161)/y(163)+y(162)/y(163)*(params(37)+(1-params(37))*(1-params(38)));
T(3) = y(104)^(-1);
T(4) = y(103)*params(40)*(1+params(41))*params(49)*y(97)*T(3);
T(5) = y(96)^(-params(42));
T(6) = T(4)*T(5);
T(7) = (y(86)/y(100))^(1+params(41));
T(8) = params(35)*params(44)*y(163)^((1+params(41))*params(43));
T(9) = params(35)*params(44)*y(163)^(params(43)-1);
T(10) = (1+y(1))/y(84);
T(11) = y(82)^(-params(45));
T(12) = y(83)^(-params(46));
T(13) = params(67)/params(93)^params(50);
T(14) = T(13)*y(114)+params(76);
T(15) = y(90)^params(50);
T(16) = y(84)^((1+params(41))*params(43));
T(17) = (1-params(44)*y(84)^(params(43)-1))/(1-params(44));
T(18) = (1+params(41))*params(43)/(params(43)-1);
T(19) = (params(43)-1)/(1+params(41)*params(43));
T(20) = y(90)^params(51);
T(21) = y(90)^(params(50)-1);
T(22) = 1/params(42);
T(23) = (y(82)/params(47))^T(22);
T(24) = (y(83)/params(48))^T(22);
T(25) = (y(104)/params(49))^((-(1+params(42)))/params(42));
T(26) = (1+params(51))*params(93)^params(51);
T(27) = params(97)^(-1);
T(28) = (params(41)+T(27))^(-1);
end
