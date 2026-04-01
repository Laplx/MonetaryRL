function [y, T, residual, g1] = dynamic_3(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(2)=exp(y(117));
  residual(1)=(y(94)/params(78))-(T(2));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(2));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
