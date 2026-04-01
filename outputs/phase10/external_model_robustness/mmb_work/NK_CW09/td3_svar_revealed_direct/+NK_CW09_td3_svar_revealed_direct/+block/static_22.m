function [y, T] = static_22(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
  T(8)=params(97)^(-1);
  y(50)=(params(41)+T(8))^(-1)*(T(8)*(y(36)+params(72)*(y(39)*params(38)*params(69)/params(72)+y(38)*(1-params(38))*params(70)/params(72)))+params(42)*y(40)+(1+params(41))*y(44));
end
