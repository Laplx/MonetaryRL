function [y, T, residual, g1] = dynamic_19(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(39)=exp(y(112));
  residual(1)=(y(89)/params(91))-(T(39));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(39));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
