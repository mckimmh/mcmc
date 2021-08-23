/* Leapfrog steps
 */
#include "leapfrog.h"
#include "log_post.h"
#include <armadillo>

void leapfrog_transform(arma::vec &x,
                        arma::vec &v,
                        LogPost &posterior,
                        double epsilon,
                        int L)
{
    arma::vec grad(x.n_elem);
    // Half-step update of velocity
    posterior.update_grad_U(x, grad);
    v -= 0.5 * epsilon * grad;
    
    // Alternate full steps for position and velocity
    for (int i = 0; i < L; ++i)
    {
        // Full step for position
        x += epsilon * v;
        
        // Full step for velocity (except at end of trajectory)
        if (i < (L-1))
        {
            posterior.update_grad_U(x, grad);
            v -= epsilon * grad;
        }
    }
    
    // Half-step update of velocity
    posterior.update_grad_U(x, grad);
    v -= 0.5 * epsilon * grad;
}

void inv_leapfrog_transform(arma::vec &x,
                            arma::vec &v,
                            LogPost &posterior,
                            double epsilon,
                            int L)
{
    v *= -1;
    leapfrog_transform(x, v, posterior, epsilon, L);
    v *= -1;
}
