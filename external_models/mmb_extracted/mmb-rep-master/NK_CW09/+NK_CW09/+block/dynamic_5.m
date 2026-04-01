function [y, T, residual, g1] = dynamic_5(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(4)=exp(y(119));
  residual(1)=(y(96)/params(88))-(T(4));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(4));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
