/* Class representing a Markov chain generated for Monte Carlo
 */
#include "log_post.h"
#include "mcmc.h"
#include "print.h"
#include <armadillo>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

MCMC::MCMC(const int burn,
           const int thin,
           const int n_samples)
    : m_burn{ burn },
    m_thin{ thin },
    m_number_samples{ n_samples },
    m_dimension{ 1 },
    m_samples_generated{ 0 }
{
}

MCMC::MCMC(LogPost posterior,
           const arma::vec &initial_state,
           const int burn,
           const int thin,
           const int n_samples)
    : m_posterior{ posterior },
    m_burn{ burn },
    m_thin{ thin },
    m_number_samples{ n_samples },
    m_dimension{ 1 },
    m_samples_generated{ 0 },
    m_initial_state{ initial_state },
    m_current{ initial_state }
{
    m_dimension = initial_state.n_elem;
    assert(m_dimension == m_posterior.get_dimension());
}

void MCMC::set_init_state(const arma::vec &initial_state)
{
    m_initial_state = initial_state;
    // Caution: changing the dimension may result in a mismatch between
    // the dimension of m_posterior and the dimension of m_inital_state
    m_dimension = initial_state.n_elem;
    m_current = initial_state;
}

void MCMC::set_burn(const int burn)
{
    m_burn = burn;
}

void MCMC::set_thin(const int thin)
{
    m_thin = thin;
}

void MCMC::set_n_samples(const int n_samples)
{
    m_number_samples = n_samples;
}

void MCMC::set_post(LogPost posterior)
{
    // Caution: changing the dimension may result in a mismatch between
    // the dimension of m_posterior and the dimension of m_inital_state
    m_posterior = posterior;
}

void MCMC::set_seed(const unsigned int s)
{
    m_gen.seed(s);
}

int MCMC::get_burn()
{
    return m_burn;
}

int MCMC::get_thin()
{
    return m_thin;
}

int MCMC::get_n_samples()
{
    return m_number_samples;
}

int MCMC::get_dimension()
{
    return m_dimension;
}

void MCMC::get_current_state(arma::vec &state)
{
    state = m_current;
}

void MCMC::get_samples(std::vector<arma::vec> &samples)
{
    assert(m_samples_generated);
    samples = m_samples;
}

void MCMC::est_moments(arma::vec &mo1_est,
                       arma::vec &mo2_est)
{
    mo1_est.zeros(m_dimension);
    mo2_est.zeros(m_dimension);
    for (std::vector<arma::vec>::iterator row = m_samples.begin();
         row != m_samples.end(); ++row)
    {
        mo1_est += *row;
        mo2_est += (*row) % (*row);
    }
    int n = m_samples.size();
    mo1_est /= n;
    mo2_est /= n;
}

void MCMC::print_current()
{
    m_current.print();
}

void MCMC::print_chain()
{
    print_vector_vec(m_samples);
}

void MCMC::print_chain(std::ofstream &file)
{
    print_vector_vec(m_samples, file);
}
