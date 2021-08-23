# Hamiltonian Monte Carlo

<em>Hamiltonian Monte Carlo</em> (HMC) is a gradient-based method for MCMC. The sampler proposes new states of the chain using approximate Hamiltonian dynamics, then uses a Metropolis-Hastings step to either accept or reject the proposed move. In order to make use of Hamiltonian dynamics, the algorithm works on an augmented state space, with an auxilliary <em>velocity</em> variable used to make proposals.

## Implementation

Class `HMC` found in `include/hmc.h` is an implementation of Hamiltonian Monte Carlo. Objects can be constructed using `HMC mc(log_post, initial_state, burn, thin, n_samples, epsilon, L)`. Here:
* `log_post` is an object of class `LogPost`, which represents the log posterior distribution that is the target distribution of the Markov Chain. This object contains information of the dimension of the target, data used in defining it, a function to evaluate the log-density and a function to evaluate the gradient of the log-density
* `initial_state` is the initial state of the Markov chain, represented as an `arma::vec`
* `burn` is the length of the burn-in period
* `thin` is the thinning used for the Markov chain
* `n_samples` is the number of thinned samples to generated
* `epsilon` is the leapfrog step-size
* `L` is the number of leapfrog steps.
Generate the Markov chain using `mc.hmc()` then print samples to a file using `mc.print_chain(file)`, for `file` an object of type `std::ofstream` which should be open, so that `file.is_open()` returns `true`.

## Example: Sampling a Bivariate Gaussian Distribution

Hamiltonian Monte Carlo is notoriously difficult to tune. One must first choose the leapfrog step-size epsilon. This can be done empirically by choosing a value of epsilon so that the acceptance rate of proposed points remains high. There tends to be a sharp stability limit after which the acceptance rate falls drastically. 

![Stability Limit of the Leapfrog Step-size](https://github.com/mckimmh/mcmc/blob/main/images/leapfrog_step_size.png)

Once the a good step-size has been found, choose the number of leapfrog steps (L) so that the autocorrelation of samples is small. Too few steps and the sampler produces a chain with random-walk like behaviour and high autocorrelation

![Autocorrelation Function when L=1](https://github.com/mckimmh/mcmc/blob/main/images/acf_L1_d1.png)

Too many leapfrog steps can generate a Markov chain with periodic behaviour, where the chain oscillates from one part of the state space to the other, so that there is a strong negative correlation between consecutive states.

![Autocorrelation Function when L=16](https://github.com/mckimmh/mcmc/blob/main/images/acf_L16_d1.png)

Just the right number of leapfrog steps, this case L=7, results in a chain with small autocorrelation.

![Autocorrelation Function when L=7](https://github.com/mckimmh/mcmc/blob/main/images/acf_L7_d1.png)

A two-dimensional traceplots of the first 20 samples of the (close to) optimally-tuned chain are shown below.

![Two-dimensional trace plot of an optimally tune Markov chain generated using HMC](https://github.com/mckimmh/mcmc/blob/main/images/trace_plot2d.png)
