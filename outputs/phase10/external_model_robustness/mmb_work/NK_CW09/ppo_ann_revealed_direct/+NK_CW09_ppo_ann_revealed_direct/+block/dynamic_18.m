function [y, T, residual, g1] = dynamic_18(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(38)=exp(y(113));
  residual(1)=(y(90)/params(93))-(T(38));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(38));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
