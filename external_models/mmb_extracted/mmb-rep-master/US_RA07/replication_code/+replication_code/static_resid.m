function [residual, T_order, T] = static_resid(y, x, params, T_order, T)
if nargin < 5
    T_order = -1;
    T = NaN(0, 1);
end
[T_order, T] = replication_code.static_resid_tt(y, x, params, T_order, T);
residual = NaN(15, 1);
    residual(1) = (y(1)) - (y(1)*params(1)+y(1)*params(2)+params(3)*y(2)+params(3)*x(1));
    residual(2) = (y(2)) - (params(8)*y(3)+(1-params(8))*(y(4)+params(9)*y(5))-y(6));
    residual(3) = (y(7)) - (y(3)*params(10));
    residual(4) = (y(4)*(1+params(4))) - (y(4)+y(4)*params(4)+y(1)*params(11)-y(1)*(1+params(4)*params(11))+y(1)*params(4)-params(12)*(y(4)-params(13)/(1-params(14))*(y(9)-params(14)*y(9))-params(15)*y(8)));
    residual(5) = (y(10)-y(7)-y(13)) - (y(3)-(y(4)+params(9)*y(5)));
    residual(6) = (y(8)) - (y(10));
    residual(7) = (y(9)*(1+params(14))) - (y(9)+params(14)*y(9)-(1-params(14))/params(13)*(y(5)-y(1)));
    residual(8) = (y(11)) - (y(11)*params(4)*(1-params(17))+y(3)*(1-params(4)*(1-params(17)))-(y(5)-y(1)));
    residual(9) = (y(13)) - (y(13)*(1-params(17))+params(17)*y(12));
    residual(10) = (y(12)) - (1/(1+params(4))*(y(12)+params(4)*y(12)+y(11)*params(18)));
    residual(11) = (y(14)) - (y(6)+params(8)*(y(7)+y(13))+(1-params(8))*y(8));
    residual(12) = (y(5)) - (y(5)*params(19)+y(1)*(1-params(19))*params(20)+y(14)*(1-params(19))*params(21)+x(2));
    residual(13) = (y(14)) - (y(9)*(1-params(22)-params(23))+y(12)*params(22)+params(23)*y(15)+y(7)*params(8)*params(7)/(params(7)-1));
    residual(14) = (y(6)) - (y(6)*params(25)+x(3));
    residual(15) = (y(15)) - (y(15)*params(26)+x(4));
end
