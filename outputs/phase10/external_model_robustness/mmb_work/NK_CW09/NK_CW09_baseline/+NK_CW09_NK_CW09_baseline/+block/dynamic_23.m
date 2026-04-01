function [y, T, residual, g1] = dynamic_23(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(43)=exp(y(127));
  residual(1)=(y(104)/params(86))-(T(43));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(43));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
