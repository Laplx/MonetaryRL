function [y, T, residual, g1] = dynamic_24(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(44)=exp(y(107));
  residual(1)=(y(83)/params(83))-(T(44));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(44));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
