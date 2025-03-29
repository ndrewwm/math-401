functions {
  // This is an embarrassingly manual implementation.
  // Stan's maintainers appear to encourage authors to avoid
  // evaluating Kronecker products as much as possible, to the
  // point where there isn't a built-in version of this operation.
  matrix kronecker_prod(matrix B, matrix K) {
    int N = rows(K);
    int N2 = N * 2;
    matrix[N, N] b11 = B[1, 1] * K;
    matrix[N, N] b12 = B[1, 2] * K;
    matrix[N, N] b21 = B[2, 1] * K;
    matrix[N, N] b22 = B[2, 2] * K;

    matrix[N2, N] L = append_row(b11, b21);
    matrix[N2, N] R = append_row(b12, b22);
    matrix[N2, N2] K2 = append_col(L, R);
    return K2;
  }
}

data {
  int<lower=1> N;   // Number of observations
  matrix[N, 2] X;   // Inputs
  matrix[N, 2] Y;   // Targets
}

transformed data {
  real delta = 1e-9;  // TODO: convert this to a parameter for estimation
  int N2 = 2*N;
  vector[N2] y = to_vector(Y);
  vector[N2] mu = rep_vector(0, N2);

  array[N] vector[2] x;
  for (n in 1:N) {
    x[n] = to_vector(row(X, n));
  }
}

parameters {
  real<lower=0> rho;    // Length-scale
}

model {
  // Alpha is fixed to 1
  matrix[N, N] K_xx = gp_exp_quad_cov(x, 1, rho);
  matrix[2, 2] B = 1/N * Y' * (K_xx \ Y);

  matrix[N2, N2] K = kronecker_prod(B, K_xx);
  for (n in 1:N2) {
    K[n, n] = K[n, n] + delta;
  }

  matrix[N2, N2] L_K = cholesky_decompose(K);

  rho ~ normal(0.8, 0.5);
  y ~ multi_normal_cholesky(mu, L_K);
}
