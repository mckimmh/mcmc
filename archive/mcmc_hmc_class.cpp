/* Class representing a Markov chain generated using Hamiltonian
 * Monte Carlo */
#include "mcmc_hmc_class.h"
#include <armadillo>

using namespace arma;
using namespace std;

HMC::HMC(LogPost log_post, const rowvec& initial_state,
         const int burn, const int thin, const int n_samples,
         const double epsilon, const int L)
{
    m_posterior = log_post;
    
    m_initial_state = initial_state;
    m_dimension = initial_state.n_elem;
    m_current = initial_state;
    m_velocity_current.set_size(m_dimension);
    m_velocity_prop.set_size(m_dimension);
    
    m_burn = burn;
    m_thin = thin;
    m_number_samples = n_samples;
    m_epsilon = epsilon;
    m_L = L;
}

HMC::HMC(const double epsilon, const int L)
{
    m_epsilon = epsilon;
    m_L = L;
}

void HMC::set_step_size_n_steps(double epsilon, int L)
{
    m_epsilon = epsilon;
    m_L = L;
}

void HMC::leapfrog_transform()
{
    // Half-step update of velocity
    m_posterior.update_grad_U(m_prop);
    m_velocity_prop -= 0.5 * m_epsilon * m_posterior.grad_U_vec;
    
    // Alternate full steps for position and velocity
    for (int i = 0; i < m_L; i++){
        // Full step for position
        m_prop += m_epsilon * m_velocity_prop;
        
        // Full step for velocity, except at end of the trajectory
        if (i < (m_L-1)){
            m_posterior.update_grad_U(m_prop);
            m_velocity_prop -= m_epsilon * m_posterior.grad_U_vec;
        }
    }
    
    // Half step of velocity at the end
    m_posterior.update_grad_U(m_prop);
    m_velocity_prop -= 0.5 * m_epsilon * m_posterior.grad_U_vec;
}

int HMC::hmc_kern()
{
    m_prop = m_current;
    
    // Gibbs update of the velocity
    // (how to make sure m_velocity has the correct dimension?)
    m_velocity_current.randn();
    
    m_velocity_prop = m_velocity_current;
    
    // Transform proposal
    leapfrog_transform();
    double current_U = m_posterior.U(m_current);
    double current_K = 0.5 * sum(dot(m_velocity_current, m_velocity_current));
    double prop_U = m_posterior.U(m_prop);
    double prop_K = 0.5 * sum(dot(m_velocity_prop, m_velocity_prop));
    
    double u = randu<double>();
    double log_accept_prob = current_U - prop_U + current_K - prop_K;
    int accept = 0;
    if (log(u) < log_accept_prob){
        m_current = m_prop;
        accept = 1;
    }
    return accept;
}

void HMC::hmc()
{
    // Burn-in period
    for (int i = 0; i < m_burn; i++){
        hmc_kern();
    }
    
    // Post-burn-in
    m_samples.set_size(m_number_samples, m_dimension);
    m_samples.row(0) = m_current;
    for (int i = 0; i < (m_number_samples-1); i++){
        // Multiple applications of the Markov kernel
        for (int j = 0; j < m_thin; j++){
            hmc_kern();
        }
        m_samples.row(i+1) = m_current;
    }
}
