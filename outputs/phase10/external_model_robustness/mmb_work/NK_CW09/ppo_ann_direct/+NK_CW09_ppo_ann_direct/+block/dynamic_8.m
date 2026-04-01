function [y, T, residual, g1] = dynamic_8(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(7)=exp(y(123));
  residual(1)=(y(100)/params(89))-(T(7));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(7));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
