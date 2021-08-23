#ifndef MCMC_H
#define MCMC_H

#include "log_post.h"
#include "print.h"
#include <armadillo>
#include <fstream>
#include <random>
#include <vector>

class MCMC
{
public:
    /* Constructor
     *
     * burn      : Burn-in period
     * thin      : Thinning interval
     * n_samples : Number of (thinned) samples to generate
     */
    MCMC(const int burn = 10000,
         const int thin = 1,
         const int n_samples = 10000);
    
    /* Constructor
     *
     * posterior     : LogPost object
     * initial_state : Initial state of the Markov chain
     * burn          : Burn-in period
     * thin          : Thinning interval
     * n_samples     : Number of (thinned) samples to generate
     */
    MCMC(LogPost posterior,
         const arma::vec &initial_state,
         const int burn = 10000,
         const int thin = 1,
         const int n_samples = 10000);
    
    // Set the initial state
    void set_init_state(const arma::vec &initial_state);
    
    // Set burn-in period
    void set_burn(const int burn);
    
    // Set thinning interval
    void set_thin(const int thin);
    
    // Set number of samples
    void set_n_samples(const int n_samples);
    
    // Set the posterior distribution of interest
    void set_post(LogPost posterior);
    
    // Set seed
    // Should only be called once, before any random numbers are generated
    void set_seed(const unsigned int s);
    
    // Get burn-in period
    int get_burn();
    
    // Get thinning interval
    int get_thin();
    
    // Get number of (thinned) samples
    int get_n_samples();
    
    // Get dimension
    int get_dimension();
    
    // Get current state
    void get_current_state(arma::vec &state);
    
    // Get samples
    void get_samples(std::vector<arma::vec> &samples);
    
    /* Estimate moments
     *
     * Estimate the first and second moments of the distribution from the
     * samples generated.
     *
     * mo1_est : estimate of the first moment
     * mo2_est : estimate of the second moment
     */
    void est_moments(arma::vec &mo1_est,
                     arma::vec &mo2_est);
    
    // Print the current state of the Markov chain
    void print_current();
    
    // Print the samples to standard output
    void print_chain();
    
    // Print the samples to a file
    void print_chain(std::ofstream& file);
    
protected:
    // Random number generator
    std::mt19937_64 m_gen;
    
    // Posterior distribution of interest
    LogPost m_posterior;
    
    // Burn in (should be a multiple of 100), thinning interval,
    // number of samples, dimension,
    // indication of whether samples have been generated.
    int m_burn, m_thin, m_number_samples, m_dimension, m_samples_generated;
    
    // Initial, current and proposal states
    arma::vec m_initial_state, m_current, m_prop;
    
    // Vector of samples
    std::vector<arma::vec> m_samples;
};

#endif
