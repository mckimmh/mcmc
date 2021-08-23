#ifndef MCMC_RWM_CLASS_H
#define MCMC_RWM_CLASS_H

#include "mcmc_class.h"
#include <armadillo>

using namespace std;
using namespace arma;

class RWM : public MCMC
{
public:
    RWM(const int burn = 10000, const int thin = 1,
        const int n_samples = 10000, const double prop_sd = 1.0);
    // Default constructor
    
    RWM(LogPost posterior, const rowvec& initial_state,
        const int burn = 10000, const int thin = 1,
        const int n_samples = 10000, const double prop_sd = 1.0);
    // Full constructor
    
    void set_prop_sd(double prop_sd);
    // Set the standard deviation of the Gaussian proposal
    
    double get_prop_sd();
    // Returns the standard deviation of the proposal;
    
    void rwm();
    // Generates a Markov chain using the Random Walk Metropolis algorithm
    
    void adapt_prop_sd();
    // Adapt the proposal standard deviation (make private later)
private:
    int rwm_sym_kern();
    // Updates m_current according to a Random Walk Metropolis symmetric kernel
    
    double m_prop_sd;
    // Standard deviation of the Gaussian proposal
};

#endif
