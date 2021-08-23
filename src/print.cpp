/* Functions for printing vectors
 */
#include "print.h"
#include <armadillo>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

void print_vector_double(std::vector<double> &v, std::ofstream &file)
{
    assert(file.is_open());
    for (std::vector<double>::iterator it = v.begin();
         it != v.end(); ++it)
    {
        file << *it << '\n';
    }
}

void print_vector_vec(std::vector<arma::vec> &v)
{
    for (std::vector<arma::vec>::iterator row = v.begin();
         row != v.end(); ++row){
        for (arma::vec::iterator col = row->begin();
             col != row->end(); ++col)
        {
            std::cout << *col << ' ';
        }
        std::cout << '\n';
    }
}

void print_vector_vec(std::vector<arma::vec> &v, std::ofstream &file)
{
    assert(file.is_open());
    for (std::vector<arma::vec>::iterator row = v.begin();
         row != v.end(); ++row){
        for (arma::vec::iterator col = row->begin();
             col != row->end(); ++col)
        {
            file << *col << ' ';
        }
        file << '\n';
    }
}
