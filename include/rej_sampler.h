/* Class representing a Rejection Sampler
 */
#ifndef REJ_SAMPLER_H
#define REJ_SAMPLER_H

#include "log_post.h"
#include "regen_dist.h"
#include <armadillo>
#include <fstream>
#include <random>
#include <vector>

class RejectionSampler
{
public:
    // Default Constructor
    RejectionSampler();
    
    /* Full constructor
     *
     * target        : Target distribution
     * prop_dist     : Proposal distribution
     * log_rej_const : Log of the rejection sampling constant C, so that
                       for all x, target(x)/(C*prop_dist(x)) <= 1.
     */
    RejectionSampler(LogPost target,
                     RegenDist prop_dist,
                     double log_rej_const = 0);
    
    /* Set seed
     *
     * s : seed
     */
    void set_seed(unsigned int s);
    
    /* Generate samples
     *
     * n : number of samples to generate
     */
    void gen(int n);
    
    /* Generate samples, adapting the rejection constant if necessary
     *
     * Warning: if the proposal distribution doesn't have heavy-enough tails,
     * then the rejection constant will be made larger and larger but the algorithm
     * will not terminate.
     *
     * n : number of samples to generate
     */
    void adapt_gen(int n);
    
    /* Get the log-rejection constant
     */
    double get_log_rej_const();
    
    /* Print samples to the console or output, or to a file
     *
     * file : file to print to
     */
    void print_samples();
    void print_samples(std::ofstream &file);
    
private:
    // RNG
    std::mt19937_64 m_gen;
    
    // Log rejection constant
    double m_log_rej_const;
    
    // Target posterior
    LogPost m_target;
    
    // Proposal distribution
    RegenDist m_prop_dist;
    
    // Store samples
    std::vector< arma::vec > m_samples;
};

#endif
