// Gaussian Process Regression for bivariate input and output data, using the
// Intrinsic Coregionalization Model (ICM).
// This program performs inference on alpha, rho, and sigma-- hyperparameters for the
// exponentiated quadratic kernel function.

functions {
  // Performs the Kronecker product B otimes A, where B \in R^{2 x 2}
  matrix kronecker_prod(matrix A, matrix B, int N, int N2) {
    matrix[N2, N2] K;
    K[1:N, 1:N] = B[1, 1] * A;
    K[1:N, (N+1):N2] = B[1, 2] * A;
    K[(N+1):N2, 1:N] = B[2, 1] * A;
    K[(N+1):N2, (N+1):N2] = B[2, 2] * A;
    return K;
  }
}

data {
  int<lower=1> N;  // Total observations
  matrix[N, 2] X;  // Inputs
  matrix[N, 2] Y;  // Targets
  matrix[2, 2] B;  // Coregionalization matrix
}

transformed data {
  int N2 = N*2;
  matrix[N, N] I_N = identity_matrix(N);
  vector[N2] y = to_vector(Y);
  vector[N2] mu = rep_vector(0, N2);

  array[N] vector[2] x;
  for (n in 1:N) {
    x[n] = to_vector(row(X, n));
  }
}

parameters {
  real<lower=0> alpha;   // Marginal standard-deviation (magnitude of the function's range)
  real<lower=0> rho;     // Length-scale
  real<lower=0> sigma1;  // Scale of the noise-term for 1st component of Y
  real<lower=0> sigma2;  // Scale of the noise-term for 2nd component of Y
}

model {
  alpha ~ std_normal();
  rho ~ inv_gamma(5, 5);
  sigma1 ~ std_normal();
  sigma2 ~ std_normal();

  matrix[2, 2] V = identity_matrix(2);
  V[1, 1] = square(sigma1);
  V[2, 2] = square(sigma2);

  matrix[N, N] k_XX = gp_exp_quad_cov(x, 1, rho);
  matrix[N2, N2] K = kronecker_prod(k_XX, B, N, N2) + kronecker_prod(I_N, V, N, N2);
  matrix[N2, N2] L_K = cholesky_decompose(K);

  y ~ multi_normal_cholesky(mu, L_K);
}
