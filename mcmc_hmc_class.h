/* Class representing a Markov chain generated using Hamiltonian
 * Monte Carlo */
#ifndef MCMC_HMC_CLASS_H
#define MCMC_HMC_CLASS_H

#include "mcmc_class.h"
#include <armadillo>

using namespace std;
using namespace arma;

class HMC : public MCMC
{
public:
    HMC(LogPost log_post, const rowvec& initial_state,
        const int burn, const int thin, const int n_samples,
        const double epsilon = 0.01, const int L = 100);
    // Constructor initializing the log posterior, initial state,
    // burn-in time (burn), thinning of samples (thin),
    // number of samples (n_samples), step-size (epsilon) and number of
    // leapfrog steps (L)
    
    HMC(const double epsilon = 0.01, const int L = 100);
    // Constructor initializing step-size (epsilon)
    // and the number of leapfrog steps (L)
    
    void set_step_size_n_steps(double epsilon, int L);
    // Get leapfrog step size (epsilon) and number of leapfrog steps (L)
    
    int hmc_kern();
    // Hamiltonian Monte Carlo Kernel
    
    void hmc();
    // Generate a Markov chain using Hamiltonian Monte Carlo
private:
    double m_epsilon;
    // Leapfrog step-size
    
    int m_L;
    // Number of leapfrog steps
    
    rowvec m_velocity_current, m_velocity_prop;
    // Vectors reprsenting the current and proposed velocities
    
    void leapfrog_transform();
    // Transforms m_prop and m_velocity according to leapfrog steps
};

#endif
