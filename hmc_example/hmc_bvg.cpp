/* HMC sampling from a Bivariate Gaussian distribution.
 *
 * Covariance matrix:
 *      1.0, 0.9
 *      0.9, 1.0
 */
#include "log_post.h"
#include "hmc.h"
#include "mcmc.h"
#include "mvg.h"
#include <armadillo>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

int main()
{
    int d = 2; // Dimension
    
    arma::vec targ_mean(d, arma::fill::zeros); // target mean
    arma::mat targ_cov({{1.0, 0.9},
                        {0.9, 1.0}});
    arma::mat targ_chol = arma::chol(targ_cov, "left");
    arma::mat targ_mean_chol = arma::join_rows(targ_mean, targ_chol);
    
    LogPost mvg(d, targ_mean_chol, mvg_ld_un, mvg_grad_ld);
    
    // Initialize from the stationary distribution
    // (are only able to do this because it's a toy example)
    arma::vec init = arma::mvnrnd(targ_mean, targ_cov);

    int burn = 0;
    int thin = 1;
    int n_samples = 1000;
    int L = 100;
    
    // Choice of step-size
    
    std::ofstream file;
    file.open("hmc_bvg_epsilon_avg_accept_rate.txt");
    if (!file.is_open()) std::cerr << "File not open\n";
    
    file << "epsilon " << "avg_accept_rate\n";
    
    for (double epsilon = 0.01; epsilon <= 1.0; epsilon+=0.01)
    {
        HMC mc(mvg, init, burn, thin, n_samples, epsilon, L);
        mc.hmc();
        
        int n_accepts = mc.get_n_accepts();
        double avg_accept_rate = (double)n_accepts/(burn + thin*n_samples);
        file << epsilon << ' ' << avg_accept_rate << '\n';
    }
    
    file.close();
    
    // Choice of Number of Leapfrog steps
    double epsilon = 0.3;
    std::string str1 = "hmc_bvg_samples_L";
    std::string str3 = ".txt";
    
    for (int L = 1; L < 21; L++)
    {
        HMC mc(mvg, init, burn, thin, n_samples, epsilon, L);
        mc.hmc();
        
        std::string str2 = std::to_string(L);
        std::string str4 = str1+str2+str3;
        file.open(str4);
        if (!file.is_open()) std::cerr << "File not open\n";
        mc.print_chain(file);
        file.close();
    }
    
    // Generate optimally tuned chain
    L = 7;
    n_samples = 100000;
    HMC mc(mvg, init, burn, thin, n_samples, epsilon, L);
    mc.hmc();
    
    file.open("hmc_bvg_samples.txt");
    if (!file.is_open()) std::cerr << "File not open\n";
    mc.print_chain(file);
    file.close();
    
    return 0;
}
