function [y, T] = dynamic_4(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
  y(103)=y(29);
  y(104)=y(47);
  y(59)=y(85)+params(5);
  y(108)=4*y(59);
  y(60)=y(83)-y(27)+params(37);
  y(57)=y(84)+params(4);
  y(106)=y(109)-y(108);
  y(110)=4*y(60);
  y(97)=y(85)+y(29)+y(104)+y(48);
  y(63)=params(37)+y(86)-y(30);
  y(62)=params(37)+y(82)-y(26);
  y(61)=params(37)+y(81)-y(25);
  y(107)=params(9)*y(67)+(1-params(9))*y(74)-y(88);
  y(105)=y(140)-y(84);
  y(102)=y(83);
  y(99)=y(97);
end
