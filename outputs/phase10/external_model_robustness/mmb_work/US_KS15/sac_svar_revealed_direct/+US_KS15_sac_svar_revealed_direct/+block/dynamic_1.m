function [y, T] = dynamic_1(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
  y(20)=params(12)*y(5)+x(1);
  y(23)=params(14)*y(8)+x(3);
  y(21)=params(13)*y(6)+x(2);
  y(27)=params(15)*y(12)+x(4);
  y(28)=params(16)*y(13)+x(5);
end
