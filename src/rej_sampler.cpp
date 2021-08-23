/* Class representing a Rejection sampler
 */

#include "log_post.h"
#include "mvg.h"
#include "print.h"
#include "regen_dist.h"
#include "rej_sampler.h"
#include <armadillo>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

RejectionSampler::RejectionSampler() : m_log_rej_const{ 0.0 }
{
}

RejectionSampler::RejectionSampler(LogPost target,
                                   RegenDist prop_dist,
                                   double log_rej_const)
    : m_log_rej_const{ log_rej_const },
    m_target{ target },
    m_prop_dist{ prop_dist }
{
    assert(target.get_dimension() == prop_dist.get_dimension());
}

void RejectionSampler::set_seed(unsigned int s)
{
    m_gen.seed(s);
}

void RejectionSampler::gen(int n)
{
    int d = m_target.get_dimension();
    arma::vec x(d, arma::fill::zeros); // Vector to store samples
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    for (int i = 0; i < n; ++i)
    {
        bool sample_generated = false;
        while (!sample_generated)
        {
            m_prop_dist.rmu(m_gen, x); // Make proposal
            double log_accept_prob =   m_target.log_dens(x)
                                     - m_prop_dist.log_dens(x)
                                     - m_log_rej_const;
            assert(log_accept_prob <= 0.0);
            double u = runif(m_gen);
            if (log(u) < log_accept_prob)
            {
                sample_generated = true;
                m_samples.push_back(x); // record sample
            }
        }
    }
}

void RejectionSampler::adapt_gen(int n)
{
    int d = m_target.get_dimension();
    arma::vec x(d, arma::fill::zeros);
    std::uniform_real_distribution<double> runif(0.0, 1.0);
    int i = 0; // number of samples generated
    double adapt_const = 0.01; // helps adaption procedure to terminate
    while (i < n)
    {
        m_prop_dist.rmu(m_gen, x);
        double log_accept_prob =   m_target.log_dens(x) - m_log_rej_const
                                 - m_prop_dist.log_dens(x);
        if (log_accept_prob > 0)
        {
            // Adapt C
            m_log_rej_const =  m_target.log_dens(x) - m_prop_dist.log_dens(x)
                             + adapt_const;
            log_accept_prob =   m_target.log_dens(x) - m_log_rej_const
                              - m_prop_dist.log_dens(x);
            // Discard samples
            i = 0;
            m_samples.clear();
        }
        double u = runif(m_gen);
        if (log(u) < log_accept_prob){
            m_samples.push_back(x);
            i++;
        }
    }
}

double RejectionSampler::get_log_rej_const()
{
    return m_log_rej_const;
}

void RejectionSampler::print_samples()
{
    print_vector_vec(m_samples);
}

void RejectionSampler::print_samples(std::ofstream &file)
{
    assert(file.is_open());
    std::vector<arma::vec>::iterator row;
    arma::vec::iterator col;
    for (row = m_samples.begin(); row != m_samples.end(); ++row){
        for (col = (*row).begin(); col != (*row).end(); ++col){
            file << *col << ' ';
        }
        file << '\n';
    }
}
