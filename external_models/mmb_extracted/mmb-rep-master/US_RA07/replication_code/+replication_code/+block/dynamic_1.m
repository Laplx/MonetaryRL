function [y, T] = dynamic_1(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
  y(21)=params(25)*y(6)+x(3);
  y(30)=params(26)*y(15)+x(4);
end
