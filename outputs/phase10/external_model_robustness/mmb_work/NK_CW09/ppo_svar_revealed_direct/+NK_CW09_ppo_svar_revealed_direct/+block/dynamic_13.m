function [y, T] = dynamic_13(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
  T(8)=params(97)^(-1);
  y(129)=(params(41)+T(8))^(-1)*(T(8)*(y(115)+params(72)*(y(118)*params(38)*params(69)/params(72)+y(117)*(1-params(38))*params(70)/params(72)))+params(42)*y(119)+(1+params(41))*y(123));
end
