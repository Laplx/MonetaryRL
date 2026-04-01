function [y, T, residual, g1] = static_25(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(34)=exp(y(43));
  residual(1)=((1+y(20))/(1+params(74)))-(T(34));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(34));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
