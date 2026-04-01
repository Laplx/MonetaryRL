function [residual, T_order, T] = dynamic_resid(y, x, params, steady_state, T_order, T)
if nargin < 6
    T_order = -1;
    T = NaN(1, 1);
end
[T_order, T] = replication_code.dynamic_resid_tt(y, x, params, steady_state, T_order, T);
residual = NaN(15, 1);
    residual(1) = (y(16)) - (params(1)*y(1)+params(2)*y(31)+params(3)*y(17)+params(3)*x(1));
    residual(2) = (y(17)) - (params(8)*y(18)+(1-params(8))*(y(19)+params(9)*y(20))-y(21));
    residual(3) = (y(22)) - (y(18)*params(10));
    residual(4) = (y(19)*(1+params(4))) - (y(4)+params(4)*y(34)+y(1)*params(11)-y(16)*(1+params(4)*params(11))+y(31)*params(4)-params(12)*(y(19)-params(13)/(1-params(14))*(y(24)-params(14)*y(9))-params(15)*y(23)));
    residual(5) = (y(25)-y(22)-y(13)) - (y(18)-(y(19)+params(9)*y(20)));
    residual(6) = (y(23)) - (y(25));
    residual(7) = (y(24)*(1+params(14))) - (params(14)*y(9)+y(39)-(1-params(14))/params(13)*(y(20)-y(31)));
    residual(8) = (y(26)) - (params(4)*(1-params(17))*y(41)+(1-params(4)*(1-params(17)))*y(33)-(y(20)-y(31)));
    residual(9) = (y(28)) - (y(13)*(1-params(17))+params(17)*y(27));
    residual(10) = (y(27)) - (T(1)*(params(4)*y(42)+y(12)+y(26)*params(18)));
    residual(11) = (y(29)) - (y(21)+params(8)*(y(22)+y(13))+(1-params(8))*y(23));
    residual(12) = (y(20)) - (params(19)*y(5)+y(16)*(1-params(19))*params(20)+y(29)*(1-params(19))*params(21)+x(2));
    residual(13) = (y(29)) - (y(24)*(1-params(22)-params(23))+y(27)*params(22)+params(23)*y(30)+y(22)*params(8)*params(7)/(params(7)-1));
    residual(14) = (y(21)) - (params(25)*y(6)+x(3));
    residual(15) = (y(30)) - (params(26)*y(15)+x(4));
end
