function [g2_v, T_order, T] = dynamic_g2(y, x, params, steady_state, T_order, T)
if nargin < 6
    T_order = -1;
    T = NaN(17, 1);
end
[T_order, T] = SW_US_fiscal.dynamic_g2_tt(y, x, params, steady_state, T_order, T);
g2_v = NaN(2, 1);
g2_v(1)=(-(x(9)*0.5*(2*T(17)+2*(y(111)+0.000000000001)*0.5*2*(y(111)+0.000000000001)*getPowerDeriv(T(1),(-0.5),1))));
g2_v(2)=(-(0.5*(1+2*(y(111)+0.000000000001)*T(17))-1));
end
