#ifndef LOG_POST_CLASS_H
#define LOG_POST_CLASS_H

#include <armadillo>
using namespace std;
using namespace arma;

class LogPost
{
public:
    LogPost(int dimension = 1);
    // Constructor - initializes dimension (default is 1).
    
    LogPost(const mat& data,
            double (*log_dens)(const rowvec& state, const mat& data),
            int dimension);
    // Constructor - initializes data, log density and dimension
    
    LogPost(const mat& data,
            double (*log_dens)(const rowvec& state, const mat& data),
            void (*grad_log_dens)(const rowvec& state,
                                  rowvec& grad,
                                  const mat& data),
            int dimension);
    // Constructor - initializes data, log density,
    // gradient of the log density and dimension.
    
    void set_data(const mat& data);
    // Sets Data
    
    void set_log_post(double (*log_dens)(const rowvec& state, const mat& data));
    // Sets the log density
    
    void set_grad_log_post(void (*grad_log_dens)(const rowvec& state,
                                                   rowvec& grad,
                                                   const mat& data));
    // Sets the gradient of the log density: a function to update grad
    
    double log_dens(rowvec& state);
    // Log density at state
    
    double U(rowvec& state);
    // Energy at state
    
    void update_grad_log_dens(rowvec& state);
    // Update grad_log_dens, the gradient of the log density at state
    
    void update_grad_U(rowvec& state);
    // Update grad_U, the gradient of the energy at state
    
    rowvec grad_log_dens_vec;
    // Gradient of the log density (how to have separate private/public? Constant pointer?)
    
    rowvec grad_U_vec;
    // Gradient of the energy
private:
    mat m_data;
    // Data
    
    int m_dimension;
    // Dimension
    
    double (*m_log_dens)(const rowvec& state, const mat& data);
    // Log density of the posterior
    
    void (*m_grad_log_dens)(const rowvec& state, rowvec& grad, const mat& data);
    // Gradient of the log density of the posterior
};

#endif
