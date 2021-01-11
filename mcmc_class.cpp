/* Class representing a Markov chain generated for Monte Carlo*/
#include "mcmc_class.h"
#include <armadillo>
#include <cmath>
#include <fstream>

using namespace std;
using namespace arma;

MCMC::MCMC(const int burn, const int thin, const int n_samples)
{
    m_burn = burn;
    m_thin = thin;
    m_number_samples = n_samples;
}

MCMC::MCMC(LogPost posterior, const rowvec& initial_state,
           const int burn, const int thin, const int n_samples)
{
    m_posterior = posterior;
    m_initial_state = initial_state;
    m_dimension = initial_state.n_elem;
    m_current = initial_state;
    m_burn = burn;
    m_thin = thin;
    m_number_samples = n_samples;
}

void MCMC::get_init_state(const rowvec& initial_state)
{
    m_initial_state = initial_state;
    m_dimension = initial_state.n_elem;
    m_current = initial_state;
}

void MCMC::get_params(const int burn, const int thin, const int n_samples)
{
    m_burn = burn;
    m_thin = thin;
    m_number_samples = n_samples;
}

void MCMC::print_params()
{
    cout << "Burn-in period: " << m_burn << "\n"
         << "Thinning: " << m_thin << "\n"
         << "Number of samples: " << m_number_samples << endl;
}

void MCMC::get_post(LogPost posterior)
{
    m_posterior = posterior;
}

void MCMC::print_current()
{
    m_current.print();
}

void MCMC::print_chain()
{
    m_samples.print();
}

void MCMC::print_chain(ofstream& file)
{
    m_samples.print(file);
}

/*
void MetropMarkovChain::rwm_adapt()
{
    for (int i = 0; i < m_number_adapts; i++){
        int accepts = 0;
        for (int j = 0; j < 100; j++){
            accepts += rwm_sym_kern();
        }
        m_prop_sd = exp(log(m_prop_sd) + accepts/100.0 - 0.234);
    }
}
*/


