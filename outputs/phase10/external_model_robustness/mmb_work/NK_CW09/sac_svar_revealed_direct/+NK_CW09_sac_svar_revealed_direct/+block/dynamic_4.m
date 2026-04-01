function [y, T, residual, g1] = dynamic_4(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(3)=exp(y(118));
  residual(1)=(y(95)/params(77))-(T(3));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(3));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
