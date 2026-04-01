function [y, T, residual, g1] = static_5(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(13))-(y(13)*params(16)+x(5));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1-params(16);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
