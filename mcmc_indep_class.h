#ifndef MCMC_INDEP_CLASS_H
#define MCMC_INDEP_CLASS_H

#include "mcmc_class.h"
#include <armadillo>

using namespace std;
using namespace arma;

class IndepSampler : public MCMC
{
public:
    IndepSampler(const vec& indep_prop_mean,
                 const mat& indep_prop_cov_mat,
                 LogPost posterior, const rowvec& initial_state,
                 const int burn = 10000, const int thin = 1,
                 const int n_samples = 10000);
    // Constructs the independent Gaussian distribution's mean and
    // covariance matrix, the posterior distribution, initial state,
    // burn-in, thinning and number of samples
    
    void set_indep_prop(const vec& indep_prop_mean,
                        const mat& indep_prop_cov_mat);
    // Set the mean and covariance matrix of the independent proposal distribution
    
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
