/* Functions for printing vectors
 */
#include <armadillo>
#include <fstream>
#include <iostream>
#include <vector>

/* Print a std::vector of doubles to a file
 *
 * v    : Vector of doubles to print
 * file : File to print to
 */
void print_vector_double(std::vector<double> &v,
                         std::ofstream &file);

/* Print a std::vector of arma::vec to the console
 *
 * v    : Vector of arma::vec to print
 */
void print_vector_vec(std::vector<arma::vec> &v);

/* Print a std::vector of arma::vec to a file
 *
 * v    : Vector of arma::vec to print
 * file : File to print to
 */
void print_vector_vec(std::vector<arma::vec> &v,
                      std::ofstream &file);
