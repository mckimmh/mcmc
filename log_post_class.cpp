/* Class representing a log posterior density*/

#include "log_post_class.h"
#include <armadillo>

LogPost::LogPost(int dimension)
{
    m_dimension = dimension;
    grad_log_dens_vec.set_size(m_dimension);
    grad_U_vec.set_size(m_dimension);
}

LogPost::LogPost(const mat& data,
                 double (*log_dens)(const rowvec& state, const mat& data),
                 int dimension)
{
    m_data = data;
    m_log_dens = log_dens;
    m_dimension = dimension;
    grad_log_dens_vec.set_size(m_dimension);
    grad_U_vec.set_size(m_dimension);
}

LogPost::LogPost(const mat& data,
                 double (*log_dens)(const rowvec& state, const mat& data),
                 void (*grad_log_dens)(const rowvec& state,
                                       rowvec& grad,
                                       const mat& data),
                 int dimension)
{
    m_data = data;
    m_log_dens = log_dens;
    m_grad_log_dens = grad_log_dens;
    m_dimension = dimension;
    grad_log_dens_vec.set_size(m_dimension);
    grad_U_vec.set_size(m_dimension);
}

void LogPost::get_data(const mat& data)
{
    m_data = data;
}

void LogPost::get_log_post(double (*log_dens)(const rowvec& state,
                                              const mat& data))
{
    m_log_dens = log_dens;
}

void LogPost::get_grad_log_post(void (*grad_log_dens)(const rowvec& state,
                                                      rowvec& grad,
                                                      const mat& data))
{
    m_grad_log_dens = grad_log_dens;
}

double LogPost::log_dens(rowvec& state)
{
    return m_log_dens(state, m_data);
}

double LogPost::U(rowvec& state)
{
    return -m_log_dens(state, m_data);
}

void LogPost::update_grad_log_dens(rowvec& state)
{
    m_grad_log_dens(state, grad_log_dens_vec, m_data);
}

void LogPost::update_grad_U(rowvec& state)
{
    m_grad_log_dens(state, grad_U_vec, m_data);
    grad_U_vec *= -1;
}
