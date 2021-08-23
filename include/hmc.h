/* Class representing a Markov chain generated using Hamiltonian
 * Monte Carlo */
#ifndef HMC_H
#define HMC_H

#include "log_post.h"
#include "mcmc.h"
#include <armadillo>

class HMC : public MCMC
{
public:
    /* Default Constructor
     *
     * burn      : The burn-in period
     * thin      : Thinning interval
     * n_samples : Number of (thinned) samples to generate
     * epsilon   : Step-size for the Leapfrog integrator
     * L         : Number of leapfrog steps.
     */
    HMC(const int burn = 10000,
        const int thin = 1,
        const int n_samples = 10000,
        const double epsilon = 0.01,
        const int L = 100);
    
    /* Constructor
     *
     * log_post      : The log posterior
     * initial_state : The initial state
     * burn          : The burn-in period
     * thin          : Thinning interval
     * n_samples     : Number of (thinned) samples to generate
     * epsilon       : Step-size for the Leapfrog integrator
     * L             : Number of leapfrog steps.
     */
    HMC(LogPost log_post,
        const arma::vec &initial_state,
        const int burn,
        const int thin,
        const int n_samples,
        const double epsilon = 0.01,
        const int L = 100);
    
    // Set leapfrog step size (epsilon)
    void set_step_size(double epsilon);
    
    // Set the number of leapfrog steps (L)
    void set_n_steps(int L);
    
    // Return leapfrog step size (epsilon)
    double get_step_size();
    
    // Return number of leapfrog steps (L)
    int get_n_steps();
    
    // Return the number of accepted moves
    int get_n_accepts();
    
    /* Adapt the leapfrog step-size
     *
     * Finds the largest value of epsilon = 2.00, 1.99, ..., 0.01,
     * for which the average acceptance rate is greater than 0.65
     * when the number of leapfrog steps is 100.
     */
    void adapt_step_size();
    
    // Generate a Markov chain using Hamiltonian Monte Carlo
    void hmc();
    
    /* Generate moment estimates
     *
     * Generate a Markov chain using Hamiltonian Monte Carlo, without saving
     * the entirity of the chain, in order to estimate the first and second
     * moments of the target distribution
     *
     * mo1_est : estimate of the first moment
     * mo2_est : estimate of the second moment
     */
    void gen_mo_est(arma::vec &mo1_est,
                    arma::vec &mo2_est);
private:
    // Leapfrog step-size
    double m_epsilon;
    
    // Number of leapfrog steps, number of accepted moves
    int m_L, m_number_accepts;
    
    // Current and proposed velocities, gradient
    arma::vec m_velocity_current, m_velocity_prop, m_grad;
    
    // Hamiltonian Monte Carlo Kernel
    // Returns an indicator of whether the proposed move was accepted or not
    int hmc_kern();
};

#endif
