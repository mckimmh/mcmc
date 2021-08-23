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
    int n_samples = 10000;

    std::ofstream file;
    
    // Proposal standard deviation "too small"
    double prop_sd = 0.1;
    RWM mc1(mvg, init, burn, thin, n_samples, prop_sd);
    mc1.rwm();
    file.open("rwm_samples_sd_small.txt");
    mc1.print_chain(file);
    file.close();
    
    //Proposal standard deviation "too large"
    prop_sd = 10;
    RWM mc2(mvg, init, burn, thin, n_samples, prop_sd);
    mc2.rwm();
    file.open("rwm_samples_sd_large.txt");
    mc2.print_chain(file);
    file.close();
    
    // Proposal standard deviation tuned
    RWM mc3(mvg, init, burn, thin, n_samples);
    mc3.adapt_prop_sd();
    std::cout << mc3.get_prop_sd() << '\n';
    mc3.rwm();
    file.open("rwm_samples_sd_tuned.txt");
    mc3.print_chain(file);
    file.close();
    
    // With thinning
    n_samples = 100000;
    thin = 30;
    RWM mc4(mvg, init, burn, thin, n_samples);
    mc4.adapt_prop_sd();
    mc4.rwm();
    file.open("rwm_samples_sd_tuned_thinned.txt");
    mc4.print_chain(file);
    file.close();
    
    return 0;
}
