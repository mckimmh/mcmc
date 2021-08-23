/* Class representing a posterior density
 */

#ifndef LOG_POST_H
#define LOG_POST_H

#include <armadillo>
#include <fstream>

class LogPost
{
public:
    /* Constructor
     *
     * dimension : Dimension of the posterior
     */
    LogPost(int dimension = 1);
    
    /* Constructor
     *
     * dimension : Dimension of the posterior
     * data      : Data passed to the posterior
     * log_dens  : Function to evaluate the log-density of the posterior at state
     *             given data
     */
    LogPost(int dimension,
            const arma::mat& data,
            double (*log_dens)(const arma::vec& state,
                               const arma::mat& data));
    
    /* Constructor
     *
     * dimension     : Dimension of the posterior
     * data          : Data passed to the posterior
     * log_dens      : Function returning the log-density of the posterior at
     *                 state given data
     * grad_log_dens : Function to compute the gradient of the log-density at state
     *                 given data then store in argument grad
     */
    LogPost(int dimension,
            const arma::mat& data,
            double (*log_dens)(const arma::vec& state,
                               const arma::mat& data),
            void (*grad_log_dens)(const arma::vec& state,
                                  arma::vec& grad,
                                  const arma::mat& data));
    
    /* Constructor
     *
     * dimension          : Dimension of the posterior
     * data               : Data passed to the posterior
     * log_dens           : Function returning the log-density of the posterior
     *                      at state given data
     * grad_log_dens      : Function to compute the gradient of the log-density at
     *                      state given data then store in argument grad
     * laplacian_log_dens : Function returning the Laplacian of the log-density
     *                    : at state given data
     */
    LogPost(int dimension,
            const arma::mat& data,
            double (*log_dens)(const arma::vec& state,
                               const arma::mat& data),
            void (*grad_log_dens)(const arma::vec& state,
                                  arma::vec& grad,
                                  const arma::mat& data),
            double (*laplacian_log_dens)(const arma::vec& state,
                                         const arma::mat& data));
    
    // Sets Data
    void set_data(const arma::mat& data);
    
    // Sets the log density
    void set_log_dens(double (*log_dens)(const arma::vec& state,
                                         const arma::mat& data));
    
    // Sets the gradient of the log density: a function to update grad
    void set_grad_log_dens(void (*grad_log_dens)(const arma::vec& state,
                                                   arma::vec& grad,
                                                   const arma::mat& data));
    
    // Sets the Laplacian of the log density
    void set_laplacian_log_dens(double (*laplacian_log_dens)
                                (const arma::vec& state,
                                 const arma::mat& data));
    
    /* Transform the density
     *
     * Allows computation of the transformed density and gradient
     *
     * la_mean : Mean of a Laplace Approximation of the target
     * la_cov  : Covariance matrix of the Laplace Approximation
     *           of the target.
     */
    void transform(const arma::vec& la_mean,
                   const arma::mat& la_cov);
    
    /* Transform the density
     *
     * Allows computation of the transformed density, gradient and Laplacian
     *
     * la_mean          : Mean of a Laplace Approximation of the target
     * la_cov           : Covariance matrix of the Laplace Approximation
     *                    of the target.
     * hessian_log_dens : For a log-density dependant on data, compute
     *                    the Hessian matrix of the log-density with respect
     *                    to the state. Store in matrix H.
     */
    void transform(const arma::vec& la_mean,
                   const arma::mat& la_cov,
                   void (*hessian_log_dens)(const arma::vec& state,
                                            arma::mat& H,
                                            const arma::mat& data));
    
    // Get the dimension
    int get_dimension();
    
    // Print data to console output
    void print_data();
    
    // Print data to file
    void print_data(std::ofstream &file);
    
    // Print transformation matrix
    void print_tf_mat();
    
    // Return indicator of whether the
    // log density / grad log density / Laplacian log density
    // has been constructed.
    int is_log_dens_constructed();
    int is_grad_log_dens_constructed();
    int is_laplacian_log_dens_constructed();
    
    // Log density at state
    double log_dens(const arma::vec& state);
    
    // Energy at state
    double U(const arma::vec& state);
    
    // Update grad_log_dens, the gradient of the log density at state
    void update_grad_log_dens(const arma::vec& state,
                              arma::vec& grad);
    
    // Update grad_U, the gradient of the energy at state
    void update_grad_U(const arma::vec& state,
                       arma::vec& grad);
    
    // Laplacian of the log density at state
    double laplacian_log_dens(const arma::vec& state);
    
    // Laplacian of the energy at state
    double laplacian_U(const arma::vec& state);
    
private:
    // Data, Laplace approximation Covariance matrix, transformation matrix
    arma::mat m_data, m_la_cov, m_tf_mat;
    
    // Laplace approximation mean
    arma::vec m_la_mean;
    
    // Dimension, indicators of whether
    // data/ m_log_dens/m_grad_log_dens/m_laplacian_log_dens
    // has been constructed, indicator of whether to transform the density
    int m_dimension, m_log_dens_constructed, m_data_constructed,
        m_grad_log_dens_constructed, m_laplacian_log_dens_constructed,
        m_transform_density;
    
    // Log density of the posterior
    double (*m_log_dens)(const arma::vec& state,
                         const arma::mat& data);
    
    // Gradient of the log density of the posterior
    void (*m_grad_log_dens)(const arma::vec& state,
                            arma::vec& grad,
                            const arma::mat& data);
    
    // Laplacian of the log density of the posterior
    double (*m_laplacian_log_dens)(const arma::vec& state,
                                   const arma::mat& data);
    
    // Hessian of the log density with respect to the state
    void (*m_hessian_log_dens)(const arma::vec& state,
                               arma::mat& H,
                               const arma::mat& data);
};

#endif
