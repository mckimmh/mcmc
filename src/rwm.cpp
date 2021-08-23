/* Class representing a Markov chain generated using
 * the Random Walk Metropolis algorithm
 */
#include "log_post.h"
#include "mcmc.h"
#include "rwm.h"
#include <armadillo>
#include <cmath>
#include <iostream>
#include <random>

RWM::RWM(const int burn,
         const int thin,
         const int n_samples) : MCMC{ burn, thin, n_samples }
{
}

RWM::RWM(LogPost posterior,
         const arma::vec &initial_state,
         const int burn,
         const int thin,
         const int n_samples,
         const double prop_sd)
    : MCMC{ posterior, initial_state, burn, thin, n_samples },
    m_prop_sd{ prop_sd }
{
    m_prop.set_size(m_dimension);
}

void RWM::set_prop_sd(double prop_sd){
    m_prop_sd = prop_sd;
}

double RWM::get_prop_sd(){
    return m_prop_sd;
}

void RWM::rwm()
{
    // Burn-in period
    for (int i = 0; i < m_burn; i++)
    {
        rwm_sym_kern();
    }
    // Post-burn-in
    m_samples.push_back(m_current);
    for (int i = 0; i < (m_number_samples-1); i++)
    {
        // Multiple applications of the Markov kernel
        for (int j = 0; j < m_thin; j++)
        {
            rwm_sym_kern();
        }
        m_samples.push_back(m_current);
    }
    m_samples_generated = 1;
}

void RWM::adapt_prop_sd()
{
    for (int i = 0; i < 100; i++)
    {
        int accepts = 0;
        for (int j = 0; j < 100; j++)
        {
            accepts += rwm_sym_kern();
        }
        m_prop_sd = exp(log(m_prop_sd) + accepts/100.0 - 0.234);
    }
}

void RWM::gen_mo_est(arma::vec &mo1_est, arma::vec &mo2_est)
{
    mo1_est.set_size(m_dimension);
    mo2_est.set_size(m_dimension);
    
    // Burn-in period
    for (int i = 0; i < m_burn; i++)
    {
        rwm_sym_kern();
    }
    
    // Post burn-in
    mo1_est = m_current;
    mo2_est = (m_current % m_current);
    for (int i = 0; i < (m_number_samples-1); ++i)
    {
        // Multiple applications of the Markov kernel
        for (int j = 0; j < m_thin; j++)
        {
            rwm_sym_kern();
        }
        mo1_est += m_current;
        mo2_est += (m_current % m_current);
    }
    
    mo1_est /= m_number_samples;
    mo2_est /= m_number_samples;
}

double RWM::est_expec_fcn(double (*fcn)(const arma::vec &state))
{
    double estimate = 0;
    // Burn-in period
    for (int i = 0; i < m_burn; ++i)
    {
        rwm_sym_kern();
    }
    // Post-burn-in
    for (int i = 0; i < m_number_samples; ++i)
    {
        // Thinning
        for (int j = 0; j < m_thin; ++j)
        {
            rwm_sym_kern();
        }
        estimate += fcn(m_current);
    }
    return estimate/m_number_samples;
}

int RWM::rwm_sym_kern()
{
    std::normal_distribution<double> rnorm(0.0, 1.0);
    for (arma::vec::iterator it = m_prop.begin(); it != m_prop.end(); ++it)
    {
        *it = rnorm(m_gen);
    }
    m_prop *= m_prop_sd;
    m_prop += m_current;
    
    int accept = 0;
    double log_accept_prob = m_posterior.log_dens(m_prop) -
                                m_posterior.log_dens(m_current);
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    double u = runif(m_gen);
    
    if (log(u) < log_accept_prob)
    {
        m_current = m_prop;
        accept = 1;
    }
    return accept;
}
