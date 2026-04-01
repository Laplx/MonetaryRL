function [T_order, T] = dynamic_resid_tt(y, x, params, steady_state, T_order, T)
if T_order >= 0
    return
end
T_order = 0;
if size(T, 1) < 16
    T = [T; NaN(16 - size(T, 1), 1)];
end
T(1) = (y(111)+0.000000000001)^2;
T(2) = y(111)+(T(1))^0.5;
T(3) = 1/(params(10)/(1-params(10)));
T(4) = 1/(1+params(41)*params(39));
T(5) = params(39)^2;
T(6) = T(5)*params(12);
T(7) = params(15)/params(39);
T(8) = (1-T(7))/(params(14)*(1+T(7)));
T(9) = (1-params(13))/(params(44)+1-params(13));
T(10) = (params(14)-1)*params(53)/(params(14)*(1+T(7)));
T(11) = 1/(1-T(7));
T(12) = T(7)/(1-T(7));
T(13) = 1/(1+params(41)*params(39)*params(21));
T(14) = (1-params(22))*(1-params(41)*params(39)*params(22))/params(22)/(1+(params(18)-1)*params(3));
T(15) = params(41)*params(39)/(1+params(41)*params(39));
T(16) = (1-params(20))*(1-params(41)*params(39)*params(20))/((1+params(41)*params(39))*params(20))*1/(1+(params(24)-1)*params(1));
end
