/* Functions to compute the log-density and simulate from,
 * a multivariate Gaussian distribution
 */
#ifndef MVG_H
#define MVG_H

#include <armadillo>
#include <random>

/* Log-denisty of a multivariate Gaussian
 *
 * Note: does not check that state, mu and L have corresponding dimensions
 *
 * state : State at which to evaluate the log-density
 * mu    : Mean
 * L     : Lower triangular matrix, the Cholesky decomposition of
 *         the covariance matrix
 */
double mvg_ld(const arma::vec &state,
              const arma::vec &mu,
              const arma::mat &L);

/* Log-denisty of a multivariate Gaussian
 *
 * state : State at which to evaluate the log-density
 * mu_L  : Mean and the lower triangular Cholesky decomposition
 *         of the covariance matrix, horizontally concatenated so
 *         that the first column is mu
 */
double mvg_ld(const arma::vec &state,
              const arma::mat &mu_L);

/* Log-density of an unnormalized multivariate Gaussian
 *
 * Note: does not check that state, mu and L have corresponding dimensions
 *
 * state : State at which to evaluate the log-density
 * mu    : Mean
 * L     : Lower triangular matrix, the Cholesky decomposition of
 *         the covariance matrix
 */
double mvg_ld_un(const arma::vec &state,
                 const arma::vec &mu,
                 const arma::mat &L);

/* Log-density of an unnormalized multivariate Gaussian
 *
 * state : State at which to evaluate the log-density
 * mu_L  : Mean and the lower triangular Cholesky decomposition
 *         of the covariance matrix, horizontally concatenated so
 *         that the first column is mu
 */
double mvg_ld_un(const arma::vec &state,
                 const arma::mat &mu_L);

/* Gradient of the log-density of a multivariate Gaussian
 *
 * Note: does not check that state, mu and L have corresponding dimensions
 *
 * state : State at which to evaluate the log-density
 * grad  : The gradient
 * mu    : Mean
 * L     : Lower triangular matrix, the Cholesky decomposition of
 *         the covariance matrix
 */
void mvg_grad_ld(const arma::vec &state,
                 arma::vec &grad,
                 const arma::vec &mu,
                 const arma::mat &L);

/* Gradient of the log-density of a multivariate Gaussian
 *
 * state : State at which to evaluate the log-density
 * grad  : The gradient
 * mu_L  : Mean and the lower triangular Cholesky decomposition
 *         of the covariance matrix, horizontally concatenated so
 *         that the first column is mu
 */
void mvg_grad_ld(const arma::vec &state,
                 arma::vec &grad,
                 const arma::mat &mu_L);

/* Laplacian of the log-denisty of a multivariate Gaussian
 *
 * Note: does not check that state, mu and L have corresponding dimensions
 *
 * state : State at which to evaluate the log-density
 * mu    : Mean
 * L     : Lower triangular matrix, the Cholesky decomposition of
 *         the covariance matrix
 */
double mvg_lap_ld(const arma::vec &state,
                  const arma::vec &mu,
                  const arma::mat &L);

/* Laplacian of the log-denisty of a multivariate Gaussian
 *
 * state : State at which to evaluate the log-density
 * mu_L  : Mean and the lower triangular Cholesky decomposition
 *         of the covariance matrix, horizontally concatenated so
 *         that the first column is mu
 */
double mvg_lap_ld(const arma::vec &state,
                  const arma::mat &mu_L);

/* Simulate from a multivariate Gaussian distribution
 *
 * Note: does not check that state, mu and L have corresponding dimensions
 *
 * generator : random number generator
 * state     : State at which to evaluate the log-density
 * mu        : Mean
 * L         : Lower triangular matrix, the Cholesky decomposition of
 *             the covariance matrix
 */
void rmvg(std::mt19937_64 &generator,
          arma::vec &state,
          const arma::vec &mu,
          const arma::mat &L);

/* Simulate from a multivariate Gaussian distribution
 *
 * generator : random number generator
 * state     : State at which to evaluate the log-density
 * mu_L      : Mean and the lower triangular Cholesky decomposition
 *             of the covariance matrix, horizontally concatenated so
 *             that the first column is mu
 *
 * Returns integer 0, an indicator of cost
 */
int rmvg(std::mt19937_64 &generator,
         arma::vec &state,
         const arma::mat &mu_L);

#endif
