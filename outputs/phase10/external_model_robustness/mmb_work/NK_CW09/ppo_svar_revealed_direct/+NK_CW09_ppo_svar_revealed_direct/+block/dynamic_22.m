function [y, T, residual, g1] = dynamic_22(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(42)=exp(y(126));
  residual(1)=(y(103)/params(87))-(T(42));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(42));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
