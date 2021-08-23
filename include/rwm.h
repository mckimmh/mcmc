#ifndef RWM_H
#define RWM_H

#include "mcmc.h"
#include <armadillo>

class RWM : public MCMC
{
public:    
    /* Default Constructor
     *
     * burn      : Burn-in period
     * thin      : Thinning interval
     * n_samples : Number of (thinned) samples to generate
     */
    RWM(const int burn = 10000,
        const int thin = 1,
        const int n_samples = 10000);
    
    /* Constructor
     *
     * posterior     : LogPost object
     * initial_state : Initial state of the Markov chain
     * burn          : Burn-in period
     * thin          : Thinning interval
     * n_samples     : Number of (thinned) samples to generate
     * prop_sd       : Standard deviation of the proposal
     */
    RWM(LogPost posterior,
        const arma::vec &initial_state,
        const int burn = 10000,
        const int thin = 1,
        const int n_samples = 10000,
        const double prop_sd = 1.0);
    
    // Set the standard deviation of the Gaussian proposal
    void set_prop_sd(double prop_sd);
    
    // Returns the standard deviation of the proposal
    double get_prop_sd();
    
    // Generates a Markov chain using the Random Walk Metropolis algorithm
    void rwm();
    
    // Adapt the proposal standard deviation (make private later)
    void adapt_prop_sd();
    
    /* Generate moment estimates
     *
     * Generate a Markov chain using Random Walk Metropolis, without recording
     * the chain, and use to estimate the first and second moments.
     *
     * mo1_est : estimate of the first moment
     * mo2_est : estimate of the second moment
     */
    void gen_mo_est(arma::vec &mo1_est,
                    arma::vec &mo2_est);
    
    // Estimate the Expectation of a scalar function using RWM
    double est_expec_fcn(double (*fcn)(const arma::vec &state));
    
private:
    // Updates m_current according to a Random Walk Metropolis
    // symmetric kernel
    int rwm_sym_kern();
    
    // Standard deviation of the Gaussian proposal
    double m_prop_sd;
};

#endif
