function [y, T, residual, g1] = static_32(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(39)=exp(y(48));
  residual(1)=(y(25)/params(86))-(T(39));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(39));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
