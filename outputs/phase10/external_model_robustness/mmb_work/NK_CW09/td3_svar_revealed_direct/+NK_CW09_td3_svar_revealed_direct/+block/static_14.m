function [y, T, residual, g1] = static_14(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(4)=exp(y(40));
  residual(1)=(y(17)/params(88))-(T(4));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(4));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
