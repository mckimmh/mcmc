/* Class Representing Importance Sampling
 */

#include "importance.h"
#include "log_post.h"
#include "mvg.h"
#include "regen_dist.h"
#include <armadillo>
#include <cassert>
#include <cmath>
#include <fstream>
#include <random>
#include <string>

ImportanceSampler::ImportanceSampler() : m_dimension{ 1 }
{
}

ImportanceSampler::ImportanceSampler(LogPost target,
                                     RegenDist importance_dist)
    : m_target{ target },
    m_importance_dist{ importance_dist }
{
    m_dimension = target.get_dimension();
    assert(m_dimension = importance_dist.get_dimension());
}

void ImportanceSampler::set_seed(unsigned int s)
{
    m_gen.seed(s);
}

void ImportanceSampler::generate(const int n)
{
    arma::vec x(m_dimension);
    for (int i = 0; i < n; ++i)
    {
        m_importance_dist.rmu(m_gen, x);
        m_samples.push_back(x);
        double w = exp(m_target.log_dens(x) - m_importance_dist.log_dens(x));
        m_weights.push_back(w);
    }
}

double ImportanceSampler::est_norm_const()
{
    double w_total = 0;
    for (std::vector<double>::iterator w = m_weights.begin();
         w != m_weights.end(); ++w)
    {
        w_total += (*w);
    }
    
    int n = m_weights.size();
    
    double Z = 0;
    if (n > 0){
        Z = w_total/n;
    }
    
    return Z;
}

int ImportanceSampler::get_dimension()
{
    return m_dimension;
}

int ImportanceSampler::get_n_samples()
{
    int n = m_weights.size();
    return n;
}

void ImportanceSampler::print_samples()
{
    for (std::vector<arma::vec>::iterator row = m_samples.begin();
         row != m_samples.end(); ++row){
        for (arma::vec::iterator col = row->begin(); col != row->end(); ++col)
        {
            std::cout << *col << ' ';
        }
        std::cout << '\n';
    }
}

void ImportanceSampler::print_samples(std::ofstream &file)
{
    assert(file.is_open());
    for (std::vector<arma::vec>::iterator row = m_samples.begin();
         row != m_samples.end(); ++row){
        for (arma::vec::iterator col = row->begin(); col != row->end(); ++col)
        {
            file << *col << ' ';
        }
        file << '\n';
    }
}

void ImportanceSampler::print_samples(std::ofstream &file,
                                      std::string file_name)
{
    if (file.is_open()) file.close();
    file.open(file_name);
    print_samples(file);
    file.close();
}

void ImportanceSampler::print_weights()
{
    for (std::vector<double>::iterator w = m_weights.begin();
         w != m_weights.end(); ++w)
    {
        std::cout << *w << '\n';
    }
}

void ImportanceSampler::print_weights(std::ofstream &file)
{
    assert(file.is_open());
    for (std::vector<double>::iterator w = m_weights.begin();
         w != m_weights.end(); ++w)
    {
        file << *w << '\n';
    }
}

void ImportanceSampler::print_weights(std::ofstream &file,
                                      std::string file_name)
{
    if (file.is_open()) file.close();
    file.open(file_name);
    print_weights(file);
    file.close();
}
