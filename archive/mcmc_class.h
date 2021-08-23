#ifndef MCMC_CLASS_H
#define MCMC_CLASS_H

#include "log_post_class.h"
#include <armadillo>
#include <fstream>
using namespace std;
using namespace arma;

class MCMC
{
public:
    MCMC(const int burn = 10000,
         const int thin = 1,
         const int n_samples = 10000);
    // Default constructor
    
    MCMC(LogPost posterior, const rowvec& initial_state,
         const int burn = 10000, const int thin = 1,
         const int n_samples = 10000);
    // Constructor setting the posterior, initial state, burn-in,
    // thinning and number of samples
    
    void set_init_state(const rowvec& initial_state);
    // Sets the initial state
    
    void set_params(const int burn, const int thin, const int n_samples);
    // Sets parameters
    
    void print_params();
    // Prints parameters
    
    void set_post(LogPost posterior);
    // Set the posterior distribution of interest
    
    void print_current();
    // Print the current state of the Markov chain
    
    void print_chain();
    // Print the samples to standard output
    
    void print_chain(ofstream& file);
    // Print the samples to a file
protected:
    LogPost m_posterior;
    // Posterior distribution of interest
    
    int m_burn, m_thin, m_number_samples, m_dimension;
    // Burn in (should be a multiple of 100), number of adaptions,
    // thinning, number of samples, dimension
    
    rowvec m_initial_state, m_current, m_prop;
    // Initial, current and proposal states
    
    mat m_samples;
    // Matrix to store samples
};

#endif
