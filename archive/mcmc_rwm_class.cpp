/* Class representing a Markov chain generated using
 * the Random Walk Metropolis algorithm
 */
#include "mcmc_rwm_class.h"
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;

RWM::RWM(const int burn, const int thin,
         const int n_samples, const double prop_sd)
{
    m_burn = burn;
    m_thin = thin;
    m_number_samples = n_samples;
    m_prop_sd = prop_sd;
}

RWM::RWM(LogPost posterior, const rowvec& initial_state,
         const int burn, const int thin,
         const int n_samples, const double prop_sd)
{
    m_posterior = posterior;
    m_initial_state = initial_state;
    m_dimension = initial_state.n_elem;
    m_current = initial_state;
    m_burn = burn;
    m_thin = thin;
    m_number_samples = n_samples;
    m_prop_sd = prop_sd;
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
    for (int i = 0; i < m_burn; i++){
        rwm_sym_kern();
    }
    // Post-burn-in
    m_samples.set_size(m_number_samples, m_dimension);
    m_samples.row(0) = m_current;
    for (int i = 0; i < (m_number_samples-1); i++){
        // Multiple applications of the Markov kernel
        for (int j = 0; j < m_thin; j++){
            rwm_sym_kern();
        }
        m_samples.row(i+1) = m_current;
    }
}

void RWM::adapt_prop_sd()
{
    for (int i = 0; i < 100; i++){
        int accepts = 0;
        for (int j = 0; j < 100; j++){
            accepts += rwm_sym_kern();
        }
        m_prop_sd = exp(log(m_prop_sd) + accepts/100.0 - 0.234);
    }
}

int RWM::rwm_sym_kern()
{
    m_prop = randn<rowvec>(m_dimension);
    m_prop *= m_prop_sd;
    m_prop += m_current;
    
    int accept = 0;
    double log_accept_prob = m_posterior.log_dens(m_prop) -
                                m_posterior.log_dens(m_current);
    double u = randu<double>();
    if (log(u) < log_accept_prob){
        m_current = m_prop;
        accept = 1;
    }
    return accept;
}
