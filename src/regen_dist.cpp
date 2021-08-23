/* Class representing a Regeneration distribution
 */
#include "mvg.h"
#include "regen_dist.h"
#include <armadillo>
#include <random>

RegenDist::RegenDist(int dimension) : m_dimension{ dimension }
{
    arma::vec mu(dimension, arma::fill::zeros);
    arma::mat L(dimension, dimension, arma::fill::eye);
    m_data = arma::join_rows(mu, L);
    m_log_dens = mvg_ld;
    m_rmu = rmvg;
}

RegenDist::RegenDist(const arma::mat &data,
                     double (*log_dens)(const arma::vec &state,
                                        const arma::mat &data),
                     int (*rmu)(std::mt19937_64 &generator,
                                arma::vec &state,
                                const arma::mat &data),
                     int dimension)
    : m_data{ data },
    m_dimension{ dimension }
{
    m_log_dens = log_dens;
    m_rmu = rmu;
}

void RegenDist::set_data(const arma::mat &data)
{
    m_data = data;
}

double RegenDist::log_dens(const arma::vec &state)
{
    return m_log_dens(state, m_data);
}

double RegenDist::U(const arma::vec &state)
{
    return -m_log_dens(state, m_data);
}

int RegenDist::rmu(std::mt19937_64 &generator,
                    arma::vec &state)
{
    return m_rmu(generator, state, m_data);
}

int RegenDist::get_dimension()
{
    return m_dimension;
}

void RegenDist::get_data(arma::mat &data)
{
    data = m_data;
}
