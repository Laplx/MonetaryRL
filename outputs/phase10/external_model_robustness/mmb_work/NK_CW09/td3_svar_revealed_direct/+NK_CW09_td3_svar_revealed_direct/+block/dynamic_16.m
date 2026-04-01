function [y, T, residual, g1] = dynamic_16(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(37)=exp(y(122));
  residual(1)=((1+y(99))/(1+params(74)))-(T(37));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(37));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
