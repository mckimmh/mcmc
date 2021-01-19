# Algorithms for MCMC

This repository contains some algorithms for MCMC: Random-Walk Metropolis,
the Independence Sampler, and Hamiltonian Monte Carlo. An example of sampling
from a 'banana' distribution is included. Algorithms are written in C++ and
use Armadillo. First compile and run file 'banana.cpp', then analyse samples
in 'banana.R'.

## banana.R
Analysis of algorithms sampling from the 'banana' distribution.

## banana.cpp
Sampling from a 'banana' distribution using Random Walk Metropolis,
the Independence Sampler and Hamiltonian Monte Carlo.

## log_post_class.cpp, log_post_class.h 
Class representing a log posterior.

## mcmc_class.cpp, mcmc_class.h
Class MCMC representing a Markov chain generated for the purposes of Monte Carlo.

## mcmc_hmc_class.cpp, mcmc_hmc_class.h
Class for Hamiltonian Monte Carlo, derived from mcmc_class.

## mcmc_indep_class.cpp, mcmc_indep_class.h
Class for the Independence sampler, derived from mcmc_class.

## mcmc_rwm_class.cpp, mcmc_rwm_class.h
Class for Random-Walk Metropolis, derived from mcmc_class.
