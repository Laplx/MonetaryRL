function [y, T, residual, g1] = static_15(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(5)=exp(y(41));
  residual(1)=(y(18)/params(80))-(T(5));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(5));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
