/* Logistic Regression Model
 *
 * Dimension agnostic functions to evaluate:
 *      log-density
 *      grad log-density
 *      Second derivative of the log-density
 *      laplacian log-density
 */
#include "log_reg.h"
#include <armadillo>
#include <cmath>

double log_reg_ld(const arma::vec& state,
                  const arma::mat& data)
{
    double sigma_squared = 400.0; // Prior variance
    double log_prior = -arma::dot(state, state) / (2*sigma_squared);
    double log_lik = 0;
    int n = data.n_rows;
    int y_col = data.n_cols - 1;
    for (int i = 0; i < n; i++){
        log_lik -= log(1.0 + exp(-data(i, y_col) *
                        arma::dot(state, data(i, arma::span(0, y_col-1)))));
    }
    return log_prior + log_lik;
}

void log_reg_grad_ld(const arma::vec &state,
                     arma::vec &grad,
                     const arma::mat &data)
{
    double sigma_squared = 400.0; // Prior variance
    double numerator, denominator, exp_term;
    int y_col = data.n_cols - 1;
    
    // Prior component of gradient (vectorized computation)
    grad = -state / (2*sigma_squared);
    
    // Likelihood component of gradient
    for (int k = 0; k < state.n_elem; k++){
        for (int i = 0; i < data.n_rows; i++){
            exp_term = exp(-data(i, y_col) *
                           arma::dot(state, data(i, arma::span(0, y_col-1))));
            numerator = data(i,y_col) * data(i,k) * exp_term;
            denominator = 1.0 + exp_term;
            grad(k) += numerator/denominator;
        }
    }
}

void log_reg_deriv2_ld(const arma::vec &state,
                       arma::vec &deriv2,
                       const arma::mat &data)
{
    double sigma_squared = 400.0; // Prior variance
    int y_col = data.n_cols - 1;
    double exp_term;
    
    for (int k = 0; k < state.n_elem; k++)
    {
        deriv2(k) = - 1.0 / sigma_squared; // Prior component
        
        for (int i = 0; i < data.n_rows; i++)
        {
            exp_term = exp(-data(i, y_col) *
                           arma::dot(state, data(i, arma::span(0, y_col-1))));
            deriv2(k) -= pow(data(i,y_col) * data(i, k), 2.0) *
                          exp_term / pow(1.0 + exp_term, 2.0);
        }
    }
}

double log_reg_lap_ld(const arma::vec &state,
                      const arma::mat &data)
{
    arma::vec deriv2(state.n_elem);
    log_reg_deriv2_ld(state, deriv2, data);
    return arma::sum(deriv2);
}

// BUG IN HERE!
void log_reg_hess_ld(const arma::vec& state,
                     arma::mat& H,
                     const arma::mat& data)
{
    double sigma_squared = 400.0; // Prior variance
    int y_col = data.n_cols - 1;
    double exp_term;
    
    H.eye();
    H *= -1.0/sigma_squared; // Prior component
    
    for (int i = 0; i < state.n_elem; ++i){
        for (int j = 0; j < state.n_elem; ++j){
            for (int k = 0; k < data.n_rows; ++k){
                exp_term = exp(-data(k, y_col) *
                               arma::dot(state, data(k, arma::span(0, y_col-1))));
                H(i,j) -= pow(data(k, y_col), 2.0) * data(k, i) * data(k, j) *
                            exp_term / pow(1.0 + exp_term, 2.0);
            }
        }
    }
}
