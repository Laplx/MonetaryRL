function [y, T, residual, g1] = static_34(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(41)=exp(y(27));
  residual(1)=(y(3)/params(82))-(T(41));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(41));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
