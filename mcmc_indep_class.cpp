/* Class representing a Markov chain generated using
 * the Independence Sampler algorithm
 */
#include "mcmc_indep_class.h"
#include <armadillo>
#include <cmath>

using namespace std;
using namespace arma;

void IndepSampler::get_indep_prop(const vec& indep_prop_mean,
                                  const mat& indep_prop_cov_mat)
{
    m_indep_prop_mean = indep_prop_mean;
    m_indep_prop_cov_mat = indep_prop_cov_mat;
    m_indep_prop_cov_mat_inverse = inv(m_indep_prop_cov_mat);
}

void IndepSampler::indep_sampler()
{
    // Burn-in period
    for (int i = 0; i < m_burn; i++){
        indep_sampler_kern();
    }
    // Post-burn-in
    m_samples.set_size(m_number_samples, m_dimension);
    m_samples.row(0) = m_current;
    for (int i = 0; i < (m_number_samples-1); i++){
        // Multiple applications of the Markov kernel
        for (int j = 0; j < m_thin; j++){
            indep_sampler_kern();
        }
        m_samples.row(i+1) = m_current;
    }
}

double IndepSampler::log_indep_prop_dens(const rowvec& state)
{
    mat z = (state-m_indep_prop_mean.t()) * m_indep_prop_cov_mat_inverse *
                (state.t()-m_indep_prop_mean);
    return -0.5 * z(0,0);
}

int IndepSampler::indep_sampler_kern()
{
    // Make proposal
    m_prop = mvnrnd(m_indep_prop_mean, m_indep_prop_cov_mat).t();
    
    double ld_indep_curr = log_indep_prop_dens(m_current);
    double ld_indep_prop = log_indep_prop_dens(m_prop);
    
    int accept = 0;
    double log_accept_prob = m_posterior.log_dens(m_prop) + ld_indep_curr -
                                m_posterior.log_dens(m_current) - ld_indep_prop;
    
    double u = randu<double>();
    if (log(u) < log_accept_prob){
        m_current = m_prop;
        accept = 1;
    }
    return accept;
}

