function [y, T] = dynamic_1(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
  y(88)=params(30)*y(32)+x(1);
  y(89)=params(31)*y(33)+x(2);
  y(90)=params(32)*y(34)+x(3)+x(1)*params(2);
  y(112)=x(8);
  y(91)=params(33)*y(35)+x(4);
  y(92)=params(34)*y(36)+x(5);
  y(65)=x(6);
  y(64)=x(7);
  y(93)=params(35)*y(37)+y(65)-params(8)*y(9);
  y(94)=params(36)*y(38)+y(64)-params(7)*y(8);
end
