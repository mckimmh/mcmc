/* Banana Distribution, Hamiltonian Monte Carlo
 *
 * g++ -Wall -larmadillo -lm banana_hmc.cpp log_post_class.cpp
 * mcmc_class.cpp mcmc_hmc_class.cpp -o banana_hmc
 */
#include "log_post_class.h"
#include "mcmc_hmc_class.h"
#include <armadillo>
#include <cmath>
#include <iostream>

#define BURN 10000
#define THIN 1
#define N_SAMPLES 100000
#define LEAPFROG_STEPS 200
#define EPSILON 1.0

using namespace std;
using namespace arma;

// Banana distribution
double log_dbanana(const rowvec& state, const mat& data);
void grad_log_dens(const rowvec& state, rowvec& grad, const mat& data);

int main()
{
    // Banana distribution (dimension 2)
    LogPost banana(2);
    banana.get_log_post(log_dbanana);
    banana.get_grad_log_post(grad_log_dens);
    
    // Hamiltonian Monte Carlo chain
    int burn = BURN;
    int thin = THIN;
    int n_samples = N_SAMPLES;
    double epsilon = EPSILON;
    int L = LEAPFROG_STEPS;
    rowvec init(2, fill::randn);
    
    // Construct hmc object
    HMC hmc_chain(banana, init, burn, thin, n_samples, epsilon, L);
    
    // HMC sampling
    hmc_chain.hmc();
    hmc_chain.print_chain();
    
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
