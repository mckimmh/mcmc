/* Declarations of functions to evaluate the log-density of
 * and simulate from a multivariate t-distribution
 */
#ifndef T_DIST_H
#define T_DIST_H

#include <armadillo>

/* Function to evaluate the log-density of a multivariate t-distribution
 *
 * x         : state at which to evaluate
 * nu        : degrees of freedom
 * mean      : mean of the t-distribution
 * Sigma     : Scale matrix
 *
 * Returns: the density of the t-distribution at x
 */
double ld_mvt(const arma::vec &x,
              const double nu,
              const arma::vec &mean,
              const arma::mat &Sigma);

/* Function to generate a sample from a multivariate t-distribution
 *
 * generator : random number generator
 * sample    : sample generated from the distribution
 * nu        : degrees of freedom
 * mean      : mean of the t-distribution
 * Sigma     : Scale matrix
 *
 * Returns: the density of the t-distribution at x
 */
void rmvt(std::mt19937_64 &generator,
          arma::vec &sample,
          const double nu,
          const arma::vec &mean,
          const arma::mat &Sigma);

#endif
