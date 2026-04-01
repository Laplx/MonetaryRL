function [y, T, residual, g1] = dynamic_2(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(1)=(1+params(51))*params(93)^params(51);
  residual(1)=(y(92))-(params(79)+y(116)/T(1));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1/T(1)));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
