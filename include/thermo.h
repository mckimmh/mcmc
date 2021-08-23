#ifndef THERMO_H
#define THERMO_H

#include "log_post.h"
#include <armadillo>
#include <random>

class ThermoInteg
{
public:
    /* Thermondynamic Integration with Geometric scaling (Constructor)
     *
     * The Expectation of the derivative of the log intermediate distribution
     * w.r.t the intermediate distribution is computed for
     * theta=1/n1, 2/n1, ..., 1 using MCMC. For theta=1/n1, the chain is started
     * at zero and uses a burn-in of length n2, after which n2 samples are generated
     * and used to estimate the expectation. Each subsequent Markov chain is
     * initialized using the final value in the previous Markov chain and no burn-in
     * period is used.
     *
     * q0 : Unnormalized density with known normalizing constant
     * q1 : Unnormalized density with unknown normalizing constant
     * z0 : Normalizing constant of q0
     * n1 : Length of vector of scaling values, evenly spaced between
     *      0 and 1: (1/n1, 2/n1, ..., 1).
     * n2 : Number of MCMC samples used to estimate the expectation of the
     *      log-density for each element in the vector of scaling values.
     */
    ThermoInteg(LogPost q0, LogPost q1, double z0, int n1, int n2);
    
    // Return an estimate of the unknown normalizing constant
    double est_norm_const();
    
    // Get vector of Expectations
    void get_expec_vec(arma::vec &expec_vec);
private:
    // Random number generator
    std::mt19937_64 m_gen;
    
    // Unnormalized density with known normalizing constant z0,
    // unnormalized density with unknown normalizing constant z1,
    LogPost m_q0, m_q1;
    
    // standard deviation of the Gaussian proposal distribution,
    // weight of mixture distribution, normalizing constant for q0
    double m_prop_sd, m_theta, m_z0;
    
    // Length of vector of scaling values, number of MCMC samples per value of
    // scaling parameter, dimension, length of burn-in
    int m_n1, m_n2, m_d, m_burn;
    
    // Vector of expectation of the derivative of the log intermediate density
    // with respect to the intermediate density
    arma::vec m_expec_vec;
    
    /* Make a proposal
     *
     * x    : current state
     * prop : proposed state
     */
    void prop_dist(const arma::vec &x, arma::vec &prop);
    
    /* Log Intermediate Density
     */
    double inter_ld(const arma::vec &x);
    
    /* Random Walk Metropolis Markov kernel with symmetric proposal
     *
     * x : state
     *
     * returns indicator of whether the proposed move was accepted
     */
    int markov_kernel(arma::vec &x);
    
    /* Adapt the standard deviation of the proposal distribution
     */
    void adapt(arma::vec &x);
    
    /* Estimate the Expectation of the derivative of the log intermediate density
     */
    double est_U(arma::vec &x);
};

#endif
