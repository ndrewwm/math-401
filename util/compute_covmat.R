#' Compute the covariance matrix for two vectors, given a kernel function.
#' 
#' @param k function
#' @param x (n x 1) vector
#' @param y (m x 1) vector
#' @return A (n x m) matrix whose entries are the kernel function for two entries of x and y
compute_covmat <- function(k, x, y) {
  n_x <- length(x)
  n_y <- length(y)
  K <- matrix(0, n_x, n_y)
  for (i in 1:n_x) {
    for (j in 1:n_y) {
      K[i, j] <- k(x[i], y[j])
    }
  }
  return(K)
}
