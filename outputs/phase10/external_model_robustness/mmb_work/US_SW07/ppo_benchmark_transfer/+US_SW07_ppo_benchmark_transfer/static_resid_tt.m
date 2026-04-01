function [T_order, T] = static_resid_tt(y, x, params, T_order, T)
if T_order >= 0
    return
end
T_order = 0;
if size(T, 1) < 11
    T = [T; NaN(11 - size(T, 1), 1)];
end
T(1) = 1/(params(10)/(1-params(10)));
T(2) = 1/(1+params(41)*params(39));
T(3) = params(39)^2;
T(4) = T(3)*params(12);
T(5) = params(15)/params(39);
T(6) = (1-T(5))/(params(14)*(1+T(5)));
T(7) = (1-params(13))/(params(44)+1-params(13));
T(8) = 1/(1+params(41)*params(39)*params(21));
T(9) = (1-params(22))*(1-params(41)*params(39)*params(22))/params(22)/(1+(params(18)-1)*params(3));
T(10) = params(41)*params(39)/(1+params(41)*params(39));
T(11) = (1-params(20))*(1-params(41)*params(39)*params(20))/((1+params(41)*params(39))*params(20))*1/(1+(params(24)-1)*params(1));
end
