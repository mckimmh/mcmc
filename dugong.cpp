/* Dugong Example
 *
 * Statistical model relating the length and age of dugongs.
 * Originating from 'Nonlinear Regression Modelling', Ratkowsky (1993),
 * analyzed in 'An Iterative Monte Carlo Method for Nonconjugate Bayesian
 * Analysis', Carlin and Gelfand (1991) and 'Adaptive Markov Chain Monte
 * Carlo through Regeneration'; Gilks, Roberts and Sahu (1998).
 *
 * Random Walk Metropolis and Independence Sampler used to sample from
 * the posterior distribution. Samples printed to files 'dugong_samples.txt'
 * and 'dugong_indep_sampler.txt' repsectively.
 *
 * g++ -Wall -std=c++11 -larmadillo -lm dugong.cpp log_post_class.cpp
 * mcmc_class.cpp mcmc_rwm_class.cpp mcmc_indep_class.cpp -o dugong
 * ./dugong
 */
#include "log_post_class.h"
#include "mcmc_rwm_class.h"
#include "mcmc_indep_class.h"
#include <armadillo>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#define BURN 100000
#define THIN 100
#define N_SAMPLES 10000

using namespace std;
using namespace arma;

double log_dens(const rowvec& state, const mat& data);

int main()
{
    vec x_vec = {1.0, 1.5, 1.5, 1.5, 2.5, 4.0, 5.0, 5.0, 7.0,
                 8.0, 8.5, 9.0, 9.5, 9.5, 10.0, 12.0, 12.0, 13.0,
                 13.0, 14.5, 15.5, 15.5, 16.5, 17.0, 22.5, 29.0, 31.5};
    
    vec y_vec = {1.8, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.35,
                 2.47, 2.19, 2.26, 2.40, 2.39, 2.41, 2.50, 2.32, 2.43,
                 2.47, 2.56, 2.65, 2.47, 2.64, 2.56, 2.70, 2.72, 2.57};
    
    mat data(27, 2);
    data.col(0) = x_vec;
    data.col(1) = y_vec;
    
    LogPost dugong;
    dugong.get_data(data);
    dugong.get_log_post(log_dens);
    
    int burn = BURN;
    int thin = THIN;
    int n_samples = N_SAMPLES;
    
    // //////////////////////
    // Random Walk Metropolis
    // //////////////////////
    
    RWM mc;
    rowvec init(3, fill::randn);
    mc.get_init_state(init);
    double proposal_sd = 0.0287; // (optimized using a separate run)
    mc.get_params(burn, thin, n_samples);
    mc.get_prop_sd(proposal_sd);
    mc.get_post(dugong);
    mc.rwm();
    
    // Print samples to file
    ofstream file;
    file.open("dugong_samples.txt");
    mc.print_chain(file);
    file.close();
    
    // ////////////////////
    // Independence Sampler
    // ////////////////////
    
    IndepSampler mc2;
    vec prop_mean = {0.97895386, -0.02414019, 1.90622305};
    mat prop_cov = {{0.0004583081, 0.0005683634, 0.0038993103},
                    {0.0005683634, 0.0039165133, 0.0009708549},
                    {0.0038993103, 0.0009708549, 0.0418873832}};
    // Start at the mean
    mc2.get_init_state(prop_mean.t());
    mc2.get_params(burn, thin, n_samples);
    mc2.get_post(dugong);
    mc2.get_indep_prop(prop_mean, prop_cov);
    mc2.indep_sampler();
    
    ofstream file2;
    file2.open("dugong_indep_sampler.txt");
    mc2.print_chain(file2);
    file2.close();

    return 0;
}

double log_dens(const rowvec& state, const mat& data)
{
    int n = data.n_rows;
    double a = 0.001;
    double sum_term = 0;
    for (int i = 0; i < n; i++){
        sum_term += pow(data(i,1) - exp(state(0)) + exp(state(1)) *
                        pow((exp(state(2)) / (1 + exp(state(2)))), data(i,0)), 2.0);
    }
    return sum(state) - (a + n/2.0)*log(2*a + sum_term) - 2*log(1.0+exp(state(2)));
}
