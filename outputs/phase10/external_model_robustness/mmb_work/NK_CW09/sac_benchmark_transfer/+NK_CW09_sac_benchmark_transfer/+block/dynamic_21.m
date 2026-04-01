function [y, T, residual, g1] = dynamic_21(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(41)=exp(y(110));
  residual(1)=(y(87)/params(84))-(T(41));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(41));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
