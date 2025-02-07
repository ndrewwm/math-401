#' Draw samples from the multivariate Gaussian distribution
#' 
#' @param n integer, the number of draws to generate
#' @param mu (d x 1) vector, contains means for each element in the space
#' @param Sigma (d x d) matrix, covariance matrix for each element in the space
#' @return A (n x d) matrix, each row is a draw, columns reflect values for each element
mvrnorm <- function(n, mu, Sigma) {
  L <- chol(Sigma) # this is L', where L'L = Sigma
  d <- length(mu)

  out <- matrix(0, nrow = n, ncol = d)
  for (i in 1:n) {
    u <- rnorm(d)
    out[i, ] <- t(L %*% u + mu)
  }
  return(out)
}
