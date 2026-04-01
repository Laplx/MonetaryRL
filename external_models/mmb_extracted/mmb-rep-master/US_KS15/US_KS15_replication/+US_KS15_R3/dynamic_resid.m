function [residual, T_order, T] = dynamic_resid(y, x, params, steady_state, T_order, T)
if nargin < 6
    T_order = -1;
    T = NaN(0, 1);
end
[T_order, T] = US_KS15_R3.dynamic_resid_tt(y, x, params, steady_state, T_order, T);
residual = NaN(15, 1);
    residual(1) = (y(16)) - ((-(params(1)+params(4)))*y(17)+params(2)*(y(3)-y(19)+y(20))+params(4)*params(5)*y(2)+y(21));
    residual(2) = (y(16)) - (params(6)*y(22)+y(23)-y(24));
    residual(3) = (y(16)) - (y(25)+y(31)-y(34));
    residual(4) = (y(24)) - (y(26)+y(27));
    residual(5) = (y(19)) - (y(34)*params(7)+y(26)*(1-params(8))*(1-params(7)*params(8))/params(8));
    residual(6) = (y(29)) - (y(22)+y(27));
    residual(7) = (y(29)) - (y(17)*(1-params(11))+y(28));
    residual(8) = (y(25)) - (params(18)*y(10)+y(19)*params(9)+y(29)*params(10)+x(6));
    residual(9) = (y(18)) - (y(34)+y(25)*(-params(17))+params(3)*y(32)-y(35));
    residual(10) = (y(30)) - (y(25)-y(34));
    residual(11) = (y(20)) - (params(12)*y(5)+x(1));
    residual(12) = (y(23)) - (params(14)*y(8)+x(3));
    residual(13) = (y(21)) - (params(13)*y(6)+x(2));
    residual(14) = (y(27)) - (params(15)*y(12)+x(4));
    residual(15) = (y(28)) - (params(16)*y(13)+x(5));
end
