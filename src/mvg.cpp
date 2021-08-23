/* Functions to compute the log-density and simulate from,
 * a multivariate Gaussian distribution
 */
#include "mvg.h"
#include <armadillo>
#include <cassert>
#include <cmath>
#include <random>

double mvg_ld(const arma::vec &state,
              const arma::vec &mu,
              const arma::mat &L)
{
    int d = state.n_elem; // Dimension
    arma::mat Sigma = L * L.t(); // Covariance matrix
    // Log normalizing constant
    double logZ = 0.5*d*log(2*M_PI) + 0.5*log(arma::det(Sigma));
    
    arma::vec y = state - mu;
    arma::vec w = arma::solve(L, y);
    
    double ss = arma::sum(w % w);
    
    return -0.5*ss - logZ;
}

double mvg_ld(const arma::vec &state,
              const arma::mat &mu_L)
{
    assert(state.n_elem == mu_L.n_rows);
    assert(mu_L.n_cols == mu_L.n_rows + 1);
    
    int d = state.n_elem;
    
    arma::vec mu = mu_L.col(0);
    arma::mat L = mu_L.cols(1, d);
    
    double ld = mvg_ld(state, mu, L);
    return ld;
}

double mvg_ld_un(const arma::vec &state,
                 const arma::vec &mu,
                 const arma::mat &L)
{
    arma::mat Sigma = L * L.t(); // Covariance matrix
    
    arma::vec y = state - mu;
    arma::vec w = arma::solve(L, y);
    
    double ss = arma::sum(w % w);
    
    return -0.5 * ss;
}

double mvg_ld_un(const arma::vec &state,
                 const arma::mat &mu_L)
{
    assert(state.n_elem == mu_L.n_rows);
    assert(mu_L.n_cols == mu_L.n_rows + 1);
    
    int d = state.n_elem;
    
    arma::vec mu = mu_L.col(0);
    arma::mat L = mu_L.cols(1, d);
    
    double ld = mvg_ld_un(state, mu, L);
    return ld;
}

void mvg_grad_ld(const arma::vec &state,
                 arma::vec &grad,
                 const arma::vec &mu,
                 const arma::mat &L)
{
    arma::vec y = state - mu;
    arma::mat Sigma = L * L.t();
    grad = arma::solve(Sigma, y);
    grad *= -1;
}

void mvg_grad_ld(const arma::vec &state,
                 arma::vec &grad,
                 const arma::mat &mu_L)
{
    assert(state.n_elem == mu_L.n_rows);
    assert(mu_L.n_cols == mu_L.n_rows + 1);
    
    int d = state.n_elem;
    
    arma::vec mu = mu_L.col(0);
    arma::mat L = mu_L.cols(1, d);
    
    mvg_grad_ld(state, grad, mu, L);
}

double mvg_lap_ld(const arma::vec &state,
                  const arma::vec &mu,
                  const arma::mat &L)
{
    arma::mat precision = arma::inv(L.t()) * arma::inv(L);
    return -arma::trace(precision);
}

double mvg_lap_ld(const arma::vec &state,
                  const arma::mat &mu_L)
{
    assert(state.n_elem == mu_L.n_rows);
    assert(mu_L.n_cols == mu_L.n_rows + 1);
    
    int d = state.n_elem;
    
    arma::vec mu = mu_L.col(0);
    arma::mat L = mu_L.cols(1, d);
    
    double lld = mvg_lap_ld(state, mu, L);
    return lld;
}

void rmvg(std::mt19937_64 &generator,
          arma::vec &state,
          const arma::vec &mu,
          const arma::mat &L)
{
    int d = state.n_elem; // Dimension
    
    // Generate an isotropic Gaussian
    arma::vec Z(d, arma::fill::zeros);
    std::normal_distribution<double> rnorm(0.0, 1.0);
    for (arma::vec::iterator it = Z.begin(); it != Z.end(); ++it)
    {
        *it = rnorm(generator);
    }
    
    // Transform
    state = (L * Z) + mu;
}

int rmvg(std::mt19937_64 &generator,
         arma::vec &state,
         const arma::mat &mu_L)
{
    assert(state.n_elem == mu_L.n_rows);
    assert(mu_L.n_cols == mu_L.n_rows + 1);
    
    int d = state.n_elem;
    
    arma::vec mu = mu_L.col(0);
    arma::mat L = mu_L.cols(1, d);
    
    rmvg(generator, state, mu, L);
    
    return 0;
}
