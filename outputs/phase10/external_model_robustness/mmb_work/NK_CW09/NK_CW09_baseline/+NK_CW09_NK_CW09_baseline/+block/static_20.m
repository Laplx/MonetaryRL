function [y, T, residual, g1] = static_20(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(12))-(y(49)+params(95)*(1-params(56))+y(12)*params(56));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1-params(56);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
