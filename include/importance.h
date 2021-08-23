/* Class Representing Importance Sampling
 */
#ifndef IMPORTANCE_H
#define IMPORTANCE_H

#include "log_post.h"
#include "regen_dist.h"
#include <armadillo>
#include <fstream>
#include <random>
#include <string>
#include <vector>

class ImportanceSampler
{
public:
    // Default Constructor
    ImportanceSampler();
    
    /* Full Constructor
     *
     * target          : Target distribution
     * importance_dist : Importance sampling distribution
     */
    ImportanceSampler(LogPost target,
                      RegenDist importance_dist);
    
    /* Set seed
     *
     * s : seed
     */
    void set_seed(unsigned int s);
    
    /* Generate importance samples
     *
     * n : number of samples to generate
     */
    void generate(const int n);
    
    // Return estimate of normalising constant based on samples generated
    double est_norm_const();
    
    // Get dimension
    int get_dimension();
    
    // Get the number of samples generated
    int get_n_samples();
    
    // Print samples
    void print_samples();
    // Print samples to file
    void print_samples(std::ofstream &file);
    // Print samples to file called file_name
    void print_samples(std::ofstream &file, std::string file_name);
    
    // Print weights
    void print_weights();
    // Print weights to file
    void print_weights(std::ofstream &file);
    // Print weights to file called file_name
    void print_weights(std::ofstream &file, std::string file_name);
    
private:
    // RNG
    std::mt19937_64 m_gen;
    
    // Target posterior
    LogPost m_target;
    
    // Importance sampling distribution
    RegenDist m_importance_dist;
    
    // Dimension, number of samples
    int m_dimension;
    
    // Samples
    std::vector<arma::vec> m_samples;
    
    // Vector storing weights
    std::vector<double> m_weights;
};

#endif
