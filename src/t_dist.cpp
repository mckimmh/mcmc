/* Function to evaluate the density of and simulate from a
 * multivariate t-distribution.
 */
#include "t_dist.h"
#include <armadillo>
#include <cmath>
#include <random>

double ld_mvt(const arma::vec &x,
              const double nu,
              const arma::vec &mean,
              const arma::mat &Sigma)
{
    int d = x.n_elem;
    arma::mat Precision = arma::inv_sympd(Sigma);
    
    double log_norm_const = log(tgamma(0.5*(nu+d)))
                            -log(tgamma(0.5*nu))
                            -0.5*d*(log(nu)+log(M_PI))
                            -0.5*log(arma::det(Sigma));

    arma::mat aux_mat = (x - mean).t() * Precision * (x - mean);
    double ld = -0.5*(nu+d)*log(1.0 + aux_mat(0,0)/nu);
    
    return log_norm_const + ld;
}

void rmvt(std::mt19937_64 &generator,
          arma::vec &sample,
          const double nu,
          const arma::vec &mean,
          const arma::mat &Sigma)
{
    int d = mean.n_elem;
    std::chi_squared_distribution<double> chi(nu);
    std::normal_distribution<double> normal(0.0, 1.0);
    double u = chi(generator); // Chi-square rv
    
    // Vector of standard normals
    arma::vec Z(d);
    for (arma::vec::iterator it = Z.begin(); it != Z.end(); ++it){
        *it = normal(generator);
    }
    
    arma::mat cholesky = arma::chol(Sigma, "lower");
    arma::vec Y = cholesky * Z;
    
    sample = mean + sqrt(nu/u) * Y;
}
