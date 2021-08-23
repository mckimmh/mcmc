#ifndef LOG_REG_V2_H
#define LOG_REG_V2_H

#include <armadillo>
#include <cmath>

// Log density
double log_reg_ld(const arma::vec& state,
                  const arma::mat& data);

// Grad log density
void log_reg_grad_ld(const arma::vec &state,
                     arma::vec &grad,
                     const arma::mat &data);

// Second derivative of the log density
// Assumes deriv2 has correct dimension
void log_reg_deriv2_ld(const arma::vec &state,
                       arma::vec &deriv2,
                       const arma::mat &data);

// Laplacian log density
double log_reg_lap_ld(const arma::vec &state,
                      const arma::mat &data);

// Hessian of the log density
void log_reg_hess_ld(const arma::vec& state,
                     arma::mat& H,
                     const arma::mat& data);

#endif
