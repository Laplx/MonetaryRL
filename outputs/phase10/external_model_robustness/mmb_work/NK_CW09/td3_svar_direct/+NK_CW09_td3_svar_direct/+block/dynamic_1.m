function [y, T] = dynamic_1(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
  y(135)=params(34)*x(11);
  y(95)=params(77)*(1-params(55))+params(55)*y(16)+x(2);
  y(94)=params(78)*(1-params(54))+params(54)*y(15)+x(1);
  y(96)=params(88)*(1-params(57))+params(57)*y(17)+x(3);
  y(97)=params(80)*(1-params(58))+params(58)*y(18)+x(4);
  y(98)=params(81)*(1-params(59))+params(59)*y(19)+x(5);
  y(114)=params(61)*y(35)+x(6);
  y(92)=params(79)*(1-params(62))+params(62)*y(13)+x(7);
  y(100)=params(89)*(1-params(63))+params(63)*y(21)+x(8);
  y(101)=params(94)*(1-params(66))+y(22)*params(66)+x(9);
end
