/* Tests for log_post_class
 *
 * g++ -Wall -std=c++11 -larmadillo -lm log_post_class_test.cpp
 * log_post_class.cpp -o test_log_post_class
 */
#include "log_post_class.h"
#include <armadillo>
#include <iostream>

using namespace std;
using namespace arma;

double log_dbanana(const rowvec& state, const mat& data);

void grad_log_dens(const rowvec& state, rowvec& grad, const mat& data);

int main(){
    
    mat data;
    LogPost banana(data, log_dbanana, grad_log_dens, 2);
    rowvec x1 = {0.0, 3.0};
    x1.print("State with minimum energy: ");
    cout << "Maximum log density: " << banana.log_dens(x1) << '\n';
    cout << "Minimum energy: " << banana.U(x1) << '\n';
    banana.update_grad_log_dens(x1);
    banana.update_grad_U(x1);
    banana.grad_log_dens_vec.print("Gradient of the log density at x1: ");
    banana.grad_U_vec.print("Gradient of the energy at x1: ");
    
    return 0;
}

double log_dbanana(const rowvec& state, const mat& data){
    return -pow(state(0), 2.0)/200 -
            pow(state(1) + 0.03*pow(state(0), 2.0) - 100*0.03, 2.0)/2;
}

void grad_log_dens(const rowvec& state, rowvec& grad, const mat& data){
    grad(0) = -0.01*state(0) - 2*0.03*state(0)*(state(1) +
                        0.03*pow(state(0), 2.0) - 100*0.03);
    grad(1) = -state(1) - 0.03*pow(state(0), 2.0) + 100*0.03;
}
