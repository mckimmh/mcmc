/* HMC sampling from a Multivariate Gaussian distribution.
 *
 * Covariance matrix:
 *      1.2, 0.4
 *      0.4, 0.8
 * Prints samples to file "bvg_hmc_samples.txt".
 */
#include "log_post.h"
#include "hmc.h"
#include "mcmc.h"
#include "mvg.h"
#include <armadillo>
#include <cmath>
#include <fstream>
#include <iostream>

int main()
{
    int d = 2; // Dimension
    
    arma::vec targ_mean(d, arma::fill::zeros); // target mean
    arma::mat targ_cov({{1.2, 0.4},
                        {0.4, 0.8}});
    arma::mat targ_chol = arma::chol(targ_cov, "left");
    arma::mat targ_mean_chol = arma::join_rows(targ_mean, targ_chol);
    
    LogPost mvg(d, targ_mean_chol, mvg_ld_un, mvg_grad_ld);
    
    // Initialize from a Gaussian distribution with variance 0.01
    arma::vec init(d, arma::fill::randn);
    init *= 0.1;

    int burn = 10000;
    int thin = 1;
    int n_samples = 100000;
    double epsilon = 0.5;
    int L = 3;
    
    HMC mc(mvg, init, burn, thin, n_samples, epsilon, L);
    
    mc.hmc();
    
    int n_accepts = mc.get_n_accepts();
    double avg_accept_rate = (double)n_accepts/(burn + thin*n_samples);
    std::cout << "Acceptance rate: " << avg_accept_rate << '\n';
    
    // Print samples to file
    std::ofstream file;
    file.open("bvg_hmc_samples.txt");
    mc.print_chain(file);
    file.close();
    
    return 0;
}
