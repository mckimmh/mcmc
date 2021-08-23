/* Random Walk Metropolis sampling of a bivariate Gaussian distribution
 * with covariance matrix
 *    1.0 0.9
 *    0.9 1.0
 */
#include "log_post.h"
#include "mcmc.h"
#include "mvg.h"
#include "print.h"
#include "rwm.h"
#include <armadillo>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

int main()
{
    int d = 2;
    
    arma::vec targ_mean(d, arma::fill::zeros); // target mean
    arma::mat targ_cov({{1.0, 0.9},
                        {0.9, 1.0}}); // Target covariance
    arma::mat targ_chol = arma::chol(targ_cov, "lower");
    arma::mat targ_mean_chol = arma::join_rows(targ_mean, targ_chol);
    LogPost mvg(d, targ_mean_chol, mvg_ld_un, mvg_grad_ld);
    
    // Initialize from the stationary distribution
    // (are only able to do this because it's a toy example)
    arma::vec init = arma::mvnrnd(targ_mean, targ_cov);

    int burn = 0;
    int thin = 1;
    int n_samples = 1000;
    
    RWM mc(mvg, init, burn, thin, n_samples);
    mc.adapt_prop_sd();
    std::cout << mc.get_prop_sd() << '\n';
    
    // Generate entire chain
    mc.rwm();
    
    return 0;
}
