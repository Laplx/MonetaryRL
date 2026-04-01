function [y, T, residual, g1] = dynamic_7(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(6)=exp(y(121));
  residual(1)=(((1-y(98))/(1-params(81)))^(-1))-(T(6));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-T(6));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
