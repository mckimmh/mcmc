/* Banana Distribution
 *
 * Sample from a 'banana' distribution via Random Walk Metropolis,
 * the Independence Sampler and Hamiltonian Monte Carlo. Print samples
 * to files "banana_rwm_samples.txt", "banana_indep_sampler.txt"
 * and "banana_hmc_samples.txt" respectively.
 *
 * g++ -Wall -std=c++11 -larmadillo -lm banana.cpp log_post_class.cpp
 * mcmc_class.cpp mcmc_rwm_class.cpp mcmc_indep_class.cpp
 * mcmc_hmc_class.cpp -o banana
 *
 * ./banana
 */
#include "log_post_class.h"
#include "mcmc_rwm_class.h"
#include "mcmc_indep_class.h"
#include "mcmc_hmc_class.h"
#include <armadillo>
#include <cmath>
#include <fstream>
#include <iostream>

#define BURN 100000
#define THIN 100
#define N_SAMPLES 100000
#define LEAPFROG_STEPS 100
#define EPSILON 1.0

using namespace std;
using namespace arma;

double log_dbanana(const rowvec& state, const mat& data);

void grad_log_dens(const rowvec& state, rowvec& grad, const mat& data);

int main()
{
    mat data;
    int dimension = 2;
    LogPost banana(data, log_dbanana, dimension);
    
    rowvec init(2, fill::randn);
    int burn = BURN;
    int thin = THIN;
    int n_samples = N_SAMPLES;
    
    // //////////////////////
    // Random Walk Metropolis
    // //////////////////////
    
    double prop_sd = 1.0;
    RWM rwm_chain(banana, init, burn, thin, n_samples, prop_sd);
    rwm_chain.adapt_prop_sd();
    rwm_chain.rwm();
    
    // Print samples to file
    ofstream file;
    file.open("banana_rwm_samples.txt");
    rwm_chain.print_chain(file);
    file.close();
    
    // ////////////////////
    // Independence Sampler
    // ////////////////////
     
    // Independent proposal mean and covariance
    vec prop_mean = {0.0, 3.0};
    mat prop_cov = {{100.0, 0.0},
                    {0.0,   1.0}};
    
    IndepSampler ismc(prop_mean, prop_cov, banana,
                      init, burn, thin, n_samples);
    ismc.indep_sampler();
    
    // Print samples to file
    file.open("banana_indep_sampler.txt");
    ismc.print_chain(file);
    file.close();
    
    // ///////////////////////
    // Hamiltonian Monte Carlo
    // ///////////////////////
    
    // Add gradient information to LogPost object
    banana.set_grad_log_post(grad_log_dens);
    
    // HMC object
    thin = 1;
    double epsilon = EPSILON;
    int L = LEAPFROG_STEPS;
    HMC hmc_chain(banana, init, burn, thin, n_samples, epsilon, L);
    
    // Generate samples
    hmc_chain.hmc();
    
    // Print to file
    file.open("banana_hmc_samples.txt");
    hmc_chain.print_chain(file);
    file.close();
    
    return 0;
}

double log_dbanana(const rowvec& state, const mat& data){
    return -pow(state(0), 2.0)/200 -
            pow(state(1) + 0.03*pow(state(0), 2.0) - 100*0.03, 2.0)/2;
}

void grad_log_dens(const rowvec& state, rowvec& grad, const mat& data){
    grad(0) = -0.01*state(0) - 2*0.03*state(0)*(state(1) +
                        0.03*pow(state(0), 2.0) - 100*0.03);
    grad(1) = -state(1) - 0.03*pow(state(0), 2.0) + 100*0.03;
}
