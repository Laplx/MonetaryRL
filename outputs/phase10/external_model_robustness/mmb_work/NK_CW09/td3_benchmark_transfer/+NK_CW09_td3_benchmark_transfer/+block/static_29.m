function [y, T, residual, g1] = static_29(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(36)=exp(y(32));
  residual(1)=(y(9)/params(85))-(T(36));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(36));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
