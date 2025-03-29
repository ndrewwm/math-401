functions {
  matrix kronecker_prod(matrix K_f, matrix K_x, int N, int N2) {
    matrix[N2, N2] K;
    K[1:N, 1:N] = K_f[1, 1] * K_x;
    K[1:N, (N+1):N2] = K_f[1, 2] * K_x;
    K[(N+1):N2, 1:N] = K_f[2, 1] * K_x;
    K[(N+1):N2, (N+1):N2] = K_f[2, 2] * K_x;
    return K;
  }
}

data {
  int<lower=1> N;
  matrix[N, 2] X;
  matrix[N, 2] Y;
  matrix[2, 2] B;
}

transformed data {
  real delta = 1e-9;  // TODO: convert this to a parameter for estimation
  int N2 = N*2;
  vector[N2] y = to_vector(Y);
  vector[N2] mu = rep_vector(0, N2);

  array[N] vector[2] x;
  for (n in 1:N) {
    x[n] = to_vector(row(X, n));
  }
}

parameters {
  real<lower=0> rho;
}

model {
  rho ~ inv_gamma(5, 5);
  matrix[N, N] K_x = gp_exp_quad_cov(x, 1, rho);
  matrix[N2, N2] K = kronecker_prod(B, K_x, N, N2);
    for (n in 1:N2) {
    K[n, n] = K[n, n] + delta;
  }

  matrix[N2, N2] L_K = cholesky_decompose(K);
  y ~ multi_normal_cholesky(mu, L_K);
}

// rho = 0.84
// # A tibble: 2 × 3
//   col   rmse_vx rmse_vy
//   <fct>   <dbl>   <dbl>
// 1 Test     3.88    8.22
// 2 Train    3.67    8.38

// rho = 1
// # A tibble: 2 × 3
//   col        rmse_vx     rmse_vy
//   <fct>        <dbl>       <dbl>
// 1 Test  2.72         3.17       
// 2 Train 0.0000000825 0.000000119
