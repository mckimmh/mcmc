/* Leapfrog steps
 */
#ifndef LEAPFROG_H
#define LEAPFROG_H

#include "log_post.h"
#include <armadillo>

/* Leapfrog transformation of state (x,v)
 *
 * x         : position
 * v         : velocity
 * posterior : LogPost object
 * epsilon   : Step-size
 * L         : number of steps
 */
void leapfrog_transform(arma::vec &x,
                        arma::vec &v,
                        LogPost &posterior,
                        double epsilon,
                        int L);

/* Inverse leapfrog transformation of state (x,v)
 *
 * x         : position
 * v         : velocity
 * posterior : LogPost object
 * epsilon   : Step-size
 * L         : number of steps
 */
void inv_leapfrog_transform(arma::vec &x,
                            arma::vec &v,
                            LogPost &posterior,
                            double epsilon,
                            int L);

#endif
