function [T_order, T] = static_resid_tt(y, x, params, T_order, T)
if T_order >= 0
    return
end
T_order = 0;
if size(T, 1) < 25
    T = [T; NaN(25 - size(T, 1), 1)];
end
T(1) = (params(37)+(1-params(37))*params(38))*y(3)/y(5)+(1-params(37))*(1-params(38))*y(4)/y(5);
T(2) = (1-params(37))*params(38)*y(3)/y(5)+y(4)/y(5)*(params(37)+(1-params(37))*(1-params(38)));
T(3) = y(25)^(-1);
T(4) = y(24)*params(40)*(1+params(41))*params(49)*y(18)*T(3);
T(5) = y(17)^(-params(42));
T(6) = T(4)*T(5);
T(7) = (y(7)/y(21))^(1+params(41));
T(8) = y(5)^((1+params(41))*params(43));
T(9) = y(5)^(params(43)-1);
T(10) = (1+y(1))/y(5);
T(11) = y(3)^(-params(45));
T(12) = y(4)^(-params(46));
T(13) = params(67)/params(93)^params(50);
T(14) = T(13)*y(35)+params(76);
T(15) = y(11)^params(50);
T(16) = (1-params(44)*T(9))/(1-params(44));
T(17) = y(11)^params(51);
T(18) = y(11)^(params(50)-1);
T(19) = 1/params(42);
T(20) = (y(3)/params(47))^T(19);
T(21) = (y(4)/params(48))^T(19);
T(22) = (y(25)/params(49))^((-(1+params(42)))/params(42));
T(23) = (1+params(51))*params(93)^params(51);
T(24) = params(97)^(-1);
T(25) = (params(41)+T(24))^(-1);
end
