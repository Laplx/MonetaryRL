function [residual, T_order, T] = static_resid(y, x, params, T_order, T)
if nargin < 5
    T_order = -1;
    T = NaN(0, 1);
end
[T_order, T] = US_KS15_empirical_taylor_rule.static_resid_tt(y, x, params, T_order, T);
residual = NaN(15, 1);
    residual(1) = (y(1)) - ((-(params(1)+params(4)))*y(2)+params(2)*(y(3)-y(4)+y(5))+y(2)*params(4)*params(5)+y(6));
    residual(2) = (y(1)) - (params(6)*y(7)+y(8)-y(9));
    residual(3) = (y(1)) - (y(1)+y(10)-y(4));
    residual(4) = (y(9)) - (y(11)+y(12));
    residual(5) = (y(4)) - (y(4)*params(7)+y(11)*(1-params(8))*(1-params(7)*params(8))/params(8));
    residual(6) = (y(14)) - (y(7)+y(12));
    residual(7) = (y(14)) - (y(2)*(1-params(11))+y(13));
    residual(8) = (y(10)) - (y(10)*params(18)+y(4)*params(9)+y(14)*params(10)+x(6));
    residual(9) = (y(3)) - (y(4)+y(10)*(-params(17))+y(2)*params(3)-y(5));
    residual(10) = (y(15)) - (y(10)-y(4));
    residual(11) = (y(5)) - (y(5)*params(12)+x(1));
    residual(12) = (y(8)) - (y(8)*params(14)+x(3));
    residual(13) = (y(6)) - (y(6)*params(13)+x(2));
    residual(14) = (y(12)) - (y(12)*params(15)+x(4));
    residual(15) = (y(13)) - (y(13)*params(16)+x(5));
end
