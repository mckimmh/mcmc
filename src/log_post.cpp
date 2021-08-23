/* Class representing a log posterior density
 *
 * Encapsulates the posterior's dimension, data, log density
 * (and energy), grad log density (and grad energy),
 * laplacian log density (and laplacian energy).
 */

#include "log_post.h"
#include <armadillo>
#include <cassert>
#include <fstream>

LogPost::LogPost(int dimension)
    : m_dimension{ dimension },
    m_log_dens_constructed{ 0 },
    m_data_constructed{ 0 },
    m_grad_log_dens_constructed{ 0 },
    m_laplacian_log_dens_constructed{ 0 },
    m_transform_density{ 0 }
{
    assert(dimension > 0);
}

LogPost::LogPost(int dimension,
                 const arma::mat& data,
                 double (*log_dens)(const arma::vec& state,
                                    const arma::mat& data))
    : m_data{ data },
    m_dimension{ dimension },
    m_log_dens_constructed{ 1 },
    m_data_constructed{ 1 },
    m_grad_log_dens_constructed{ 0 },
    m_laplacian_log_dens_constructed{ 0 },
    m_transform_density{ 0 }
{
    assert(dimension > 0);
    m_log_dens = log_dens;
}

LogPost::LogPost(int dimension,
                 const arma::mat& data,
                 double (*log_dens)(const arma::vec& state,
                                    const arma::mat& data),
                 void (*grad_log_dens)(const arma::vec& state,
                                       arma::vec& grad,
                                       const arma::mat& data))
    : m_data{ data },
    m_dimension{ dimension },
    m_log_dens_constructed{ 1 },
    m_data_constructed{ 1 },
    m_grad_log_dens_constructed{ 1 },
    m_laplacian_log_dens_constructed{ 0 },
    m_transform_density{ 0 }
{
    assert(dimension > 0);
    m_log_dens = log_dens;
    m_grad_log_dens = grad_log_dens;
}

LogPost::LogPost(int dimension,
                 const arma::mat& data,
                 double (*log_dens)(const arma::vec& state,
                                    const arma::mat& data),
                 void (*grad_log_dens)(const arma::vec& state,
                                       arma::vec& grad,
                                       const arma::mat& data),
                 double (*laplacian_log_dens)(const arma::vec& state,
                                              const arma::mat& data))
    : m_data{ data },
    m_dimension{ dimension },
    m_log_dens_constructed{ 1 },
    m_data_constructed{ 1 },
    m_grad_log_dens_constructed{ 1 },
    m_laplacian_log_dens_constructed{ 1 },
    m_transform_density{ 0 }
{
    assert(dimension > 0);
    m_log_dens = log_dens;
    m_grad_log_dens = grad_log_dens;
    m_laplacian_log_dens = laplacian_log_dens;
}

void LogPost::set_data(const arma::mat& data)
{
    m_data = data;
    m_data_constructed = 1;
}

void LogPost::set_log_dens(double (*log_dens)(const arma::vec& state,
                                              const arma::mat& data))
{
    m_log_dens = log_dens;
    m_log_dens_constructed = 1;
}

void LogPost::set_grad_log_dens(void (*grad_log_dens)(const arma::vec& state,
                                                      arma::vec& grad,
                                                      const arma::mat& data))
{
    m_grad_log_dens = grad_log_dens;
    m_grad_log_dens_constructed = 1;
}

void LogPost::set_laplacian_log_dens(double (*laplacian_log_dens)
                                      (const arma::vec& state,
                                       const arma::mat& data))
{
    m_laplacian_log_dens = laplacian_log_dens;
    m_laplacian_log_dens_constructed = 1;
}

void LogPost::transform(const arma::vec& la_mean,
                        const arma::mat& la_cov)
{
    // Check la_mean and la_cov have the correct dimension
    if ((int)la_mean.n_elem != m_dimension){
        std::cerr << "la_mean has incorrect dimension!" << std::endl;
    }
    if (((int)la_cov.n_rows != m_dimension) ||
        ((int)la_cov.n_cols != m_dimension)){
        std::cerr << "la_cov has incorrect dimension!" << std::endl;
    }
    
    // Check la_cov is symmetric positive definite
    if (!la_cov.is_sympd()){
        std::cerr << "la_cov is not symmetric positive definite!" << std::endl;
    }
    
    // Record Laplace Approximation used
    m_la_mean = la_mean;
    m_la_cov = la_cov;
    
    // Eigenvalue decomposition of the Laplace Approximation's
    // covariance matrix
    arma::vec eigval;
    arma::mat eigvec;
    eig_sym(eigval, eigvec, m_la_cov);
    
    arma::mat Lambda(m_dimension, m_dimension, arma::fill::zeros);
    for (int i = 0; i < m_dimension; i++)
    {
        Lambda(i,i) = sqrt(eigval(i));
    }
    
    m_tf_mat = eigvec * Lambda;
    
    // Indicate density is transformed
    m_transform_density = 1;
}

void LogPost::transform(const arma::vec& la_mean,
                        const arma::mat& la_cov,
                        void (*hessian_log_dens)(const arma::vec& state,
                                                 arma::mat& H,
                                                 const arma::mat& data))
{
    transform(la_mean, la_cov);
    m_hessian_log_dens = hessian_log_dens;
}

int LogPost::get_dimension()
{
    return m_dimension;
}

void LogPost::print_data()
{
    if (m_data_constructed){
        m_data.print();
    } else {
        std::cerr << "Data hasn't been constructed yet\n";
    }
}

void LogPost::print_data(std::ofstream &file)
{
    m_data.print(file);
}

void LogPost::print_tf_mat()
{
    if (m_transform_density){
        m_tf_mat.print();
    } else {
        std::cerr << "Density hasn't been transformed yet\n";
    }
}

int LogPost::is_log_dens_constructed()
{
    return m_log_dens_constructed;
}

int LogPost::is_grad_log_dens_constructed()
{
    return m_grad_log_dens_constructed;
}

int LogPost::is_laplacian_log_dens_constructed()
{
    return m_laplacian_log_dens_constructed;
}

double LogPost::log_dens(const arma::vec& state)
{
    double ld;
    if (m_transform_density){
        arma::vec orig_state;
        orig_state = m_la_mean + m_tf_mat * state;
        ld = m_log_dens(orig_state, m_data);
    } else {
        ld = m_log_dens(state, m_data);
    }
    return ld;
}

double LogPost::U(const arma::vec& state)
{
    return -log_dens(state);
}

void LogPost::update_grad_log_dens(const arma::vec& state,
                                   arma::vec& grad)
{
    if (m_transform_density){
        arma::vec orig_state;
        orig_state = m_la_mean + m_tf_mat * state;
        
        arma::vec aux_grad(m_dimension);
        m_grad_log_dens(orig_state, aux_grad, m_data);
        grad = m_tf_mat.t() * aux_grad;
    } else {
        m_grad_log_dens(state, grad, m_data);
    }
}

void LogPost::update_grad_U(const arma::vec& state,
                            arma::vec& grad)
{
    update_grad_log_dens(state, grad);
    grad *= -1;
}

double LogPost::laplacian_log_dens(const arma::vec& state)
{
    double lld = 0; // (Initialize for now to silence warning)
    if (m_transform_density){
        // Compute the Hessian
        arma::mat H(m_dimension, m_dimension);
        arma::vec orig_state;
        orig_state = m_la_mean + m_tf_mat * state;
        m_hessian_log_dens(orig_state, H, m_data);
        
        // Compute the Laplacian
        arma::vec col_i(m_dimension);
        arma::mat aux_mat;
        for (int i = 0; i < m_dimension; ++i)
        {
            col_i = m_tf_mat.col(i);
            aux_mat = col_i.t() * H * col_i;
            lld += aux_mat(0,0);
        }
    } else {
        lld = m_laplacian_log_dens(state, m_data);
    }
    return lld;
}

double LogPost::laplacian_U(const arma::vec& state)
{
    return -laplacian_log_dens(state);
}
