#ifndef MCMC_INDEP_CLASS_H
#define MCMC_INDEP_CLASS_H

#include "mcmc_class.h"
#include <armadillo>

using namespace std;
using namespace arma;

class IndepSampler : public MCMC
{
public:
    void get_indep_prop(const vec& indep_prop_mean,
                        const mat& indep_prop_cov_mat);
    // Get the mean and covariance matrix of the independent proposal distribution
    
    void indep_sampler();
    // Generate a Markov chain using the Independence sampler algorithm
    
private:
    vec m_indep_prop_mean;
    // Mean of the independent proposal
    
    mat m_indep_prop_cov_mat, m_indep_prop_cov_mat_inverse;
    // Covariance matrix of the independent proposal
    
    double log_indep_prop_dens(const rowvec& state);
    // Evaluate (up to an additive constant) the log density of the
    // independent proposal
    
    int indep_sampler_kern();
    // Updates m_current according to an Independent Sampler kernel
};

#endif
