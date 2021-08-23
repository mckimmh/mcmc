/* Class representing a Markov chain generated using Hamiltonian
 * Monte Carlo */
#include "hmc.h"
#include "leapfrog.h"
#include "log_post.h"
#include "mcmc.h"
#include "print.h"
#include <armadillo>
#include <cassert>
#include <iostream>
#include <random>

HMC::HMC(const int burn,
         const int thin,
         const int n_samples,
         const double epsilon,
         const int L)
: MCMC{ burn, thin, n_samples },
m_epsilon{ epsilon },
m_L{ L },
m_number_accepts{ 0 }
{
}

HMC::HMC(LogPost log_post,
         const arma::vec &initial_state,
         const int burn,
         const int thin,
         const int n_samples,
         const double epsilon,
         const int L)
: MCMC{ log_post, initial_state, burn, thin, n_samples },
m_epsilon{ epsilon },
m_L{ L },
m_number_accepts{ 0 }
{
    assert(m_posterior.is_grad_log_dens_constructed());
    m_velocity_current.set_size(m_dimension);
    m_velocity_prop.set_size(m_dimension);
}

void HMC::set_step_size(double epsilon)
{
    m_epsilon = epsilon;
}

void HMC::set_n_steps(int L)
{
    m_L = L;
}

double HMC::get_step_size()
{
    return m_epsilon;
}

int HMC::get_n_steps()
{
    return m_L;
}

int HMC::get_n_accepts()
{
    return m_number_accepts;
}

void HMC::adapt_step_size()
{
    // Check gradient information available
    if (!m_posterior.is_grad_log_dens_constructed()){
        std::cerr << "LogPost object doesn't contain gradient information!\n";
    }
    
    m_epsilon = 2.00;
    int originaL = m_L; // Save the orignal choice of number of steps
    m_L = 100; // Make the number of steps large
    int epsilon_sufficiently_small = 0;
    
    while ((m_epsilon > 0.015) && !epsilon_sufficiently_small)
    {
        int n_accepts = 0;
        for (int i = 0; i < 1000; ++i) n_accepts += hmc_kern();
        double avg_accept = n_accepts/1000.0;
        if (avg_accept > 0.65){
            epsilon_sufficiently_small = 1;
        } else {
            m_epsilon -= 0.01;
        }
    }
    m_L = originaL; // Revert to orignal choice of nuber of steps
}

void HMC::hmc()
{
    // Check gradient information available
    if (!m_posterior.is_grad_log_dens_constructed()){
        std::cerr << "LogPost object doesn't contain gradient information!\n";
    }
    
    // Burn-in period
    for (int i = 0; i < m_burn; i++) m_number_accepts += hmc_kern();
    
    // Post-burn-in
    m_samples.push_back(m_current);
    for (int i = 0; i < (m_number_samples-1); i++)
    {
        // Multiple applications of the Markov kernel
        for (int j = 0; j < m_thin; j++)
        {
            m_number_accepts += hmc_kern();
        }
        m_samples.push_back(m_current);
    }
    m_samples_generated = 1;
}

void HMC::gen_mo_est(arma::vec &mo1_est,
                     arma::vec &mo2_est)
{
    // Check gradient information available
    if (!m_posterior.is_grad_log_dens_constructed()){
        std::cerr << "LogPost object doesn't contain gradient information!\n";
    }
    
    // Make sure moment estimates have correct dimension
    mo1_est.zeros(m_dimension);
    mo2_est.zeros(m_dimension);
    
    // Burn-in period
    for (int i = 0; i < m_burn; ++i) m_number_accepts += hmc_kern();
    
    // Post-burn-in
    for (int i = 0; i < m_number_samples; ++i)
    {
        // Multiple applications of the Markov kernel
        for (int j = 0; j < m_thin; ++j)
        {
            m_number_accepts += hmc_kern();
        }
        mo1_est += m_current;
        mo2_est += m_current % m_current;
    }
    
    mo1_est /= m_number_samples;
    mo2_est /= m_number_samples;
}

int HMC::hmc_kern()
{
    m_prop = m_current;
    
    // Gibbs update of the velocity
    std::normal_distribution<double> rnorm(0.0, 1.0);
    for (arma::vec::iterator it = m_velocity_current.begin();
         it != m_velocity_current.end(); ++it)
    {
        *it = rnorm(m_gen);
    }
    
    m_velocity_prop = m_velocity_current;
    leapfrog_transform(m_prop, m_velocity_prop, m_posterior, m_epsilon, m_L);
    
    // Compute acceptance probability
    double current_U = m_posterior.U(m_current);
    double current_K = 0.5 * arma::sum(arma::dot(m_velocity_current, m_velocity_current));
    double prop_U = m_posterior.U(m_prop);
    double prop_K = 0.5 * arma::sum(arma::dot(m_velocity_prop, m_velocity_prop));
    double log_accept_prob = current_U - prop_U + current_K - prop_K;
    
    // Accept or reject
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    double u = runif(m_gen);
    
    int accept = 0;
    if (log(u) < log_accept_prob)
    {
        m_current = m_prop;
        accept = 1;
    }
    
    return accept;
}
