function [y, T, residual, g1] = static_17(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(7)=exp(y(44));
  residual(1)=(y(21)/params(89))-(T(7));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(7));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
