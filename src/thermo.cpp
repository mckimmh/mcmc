/* Thermodynamic Integration with Geometric Path and Random Walk Metropolis Sampling
 */
#include "log_post.h"
#include "thermo.h"
#include <armadillo>
#include <cassert>
#include <cmath>
#include <random>

ThermoInteg::ThermoInteg(LogPost q0, LogPost q1, double z0, int n1, int n2)
   : m_q0{ q0 }, m_q1{ q1 }, m_prop_sd{ 1.0 }, m_theta{ 0.0 },
     m_z0{ z0 }, m_n1{ n1 }, m_n2{ n2 }, m_burn{ n2 }
{
    assert(q0.get_dimension() == q1.get_dimension());
    m_d = q0.get_dimension();
    m_expec_vec.zeros(n1);
}

double ThermoInteg::est_norm_const()
{
    double theta_incr = 1.0/m_n1;
    m_theta = theta_incr;
    arma::vec x(m_d, arma::fill::zeros); // initialize at 0
    adapt(x);
    m_expec_vec(0) = est_U(x);
    m_burn = 0; // no longer need burn-in period
    
    for (int i = 1; i < m_n1; ++i) // zero-th iteration already done
    {
        m_theta += theta_incr;
        m_expec_vec(i) = est_U(x);
    }
    
    return exp(arma::mean(m_expec_vec) + log(m_z0));
}

void ThermoInteg::get_expec_vec(arma::vec &expec_vec)
{
    expec_vec = m_expec_vec;
}

void ThermoInteg::prop_dist(const arma::vec &x, arma::vec &prop)
{
    std::normal_distribution<double> rnorm(0.0, 1.0);
    for (arma::vec::iterator it = prop.begin(); it != prop.end(); ++it)
    {
        *it = rnorm(m_gen);
    }
    prop *= m_prop_sd;
    prop += x;
}

double ThermoInteg::inter_ld(const arma::vec &x)
{
    return (1-m_theta)*m_q0.log_dens(x) + m_theta*m_q1.log_dens(x);
}

int ThermoInteg::markov_kernel(arma::vec &x)
{
    arma::vec prop(m_d);
    prop_dist(x, prop);
    int accept = 0;
    double log_accept_prob = inter_ld(prop) - inter_ld(x);
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    double u = runif(m_gen);
    if (log(u) < log_accept_prob)
    {
        x = prop;
        accept = 1;
    }
    return accept;
}

void ThermoInteg::adapt(arma::vec &x)
{
    for (int i = 0; i < 100; ++i)
    {
        int accepts = 0;
        for (int j = 0; j < 100; ++j)
        {
            accepts += markov_kernel(x);
        }
        m_prop_sd = exp(log(m_prop_sd) + accepts/100.0 - 0.234);
    }
}

double ThermoInteg::est_U(arma::vec &x)
{
    double expec_est = 0;
    for (int i = 0; i < m_burn; ++i) markov_kernel(x); // burn-in
    for (int i = 0; i < m_n2; ++i)
    {
        markov_kernel(x);
        expec_est += m_q1.log_dens(x) - m_q0.log_dens(x);
    }
    return expec_est/m_n2;
}
