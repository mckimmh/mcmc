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

We consider using HMC to sample from a bivariate zero-mean Gaussian distribution. Let the variances of both components be 1 and let the covariance between variables be 0.9. Though this is a low-dimensional example, the strong correlation between components makes it an interesting test case. When Hamiltonian dynamics are simulated exactly, the Hamiltonian (the energy corresponding to the augmented target distribution) is preserved exactly. In practice, the dynamics must be simulated approximately using a numerical integrator. The most popular integrator is the Leapfrog method, which alternates between updating the velocity and the position variables. Tuning HMC is notoriously difficult; it amounts to choosing a suitable leapfrog step-size (epsilon) and number of steps (L).

First use a large number of leapfrog steps to help determine a suitable step-size. Epsilon should be chosen to be as large as possible (to minimize computational cost), whilst ensuring the Hamiltonian trajectories remain stable. A low acceptance rate of proposed moves indicates that trajectories are unstable. The plot below shows the average acceptance rate for L=100 and epsilon = 0.01, 0.02, ..., 1.00. There is a sharp deterioration in stability after rouglhy epsilon = 0.5. We choose to use epsilon = 0.30.

![Stability Limit of the Leapfrog Step-size](https://github.com/mckimmh/mcmc/blob/main/images/leapfrog_step_size.png)

Once the step-size is fixed, choose the number of leapfrog steps so that the correlation between consecutive states in the Markov chain is as small as possible. In other words, we want the chain's autocorrelation to be small. The plots below show the Auto-correlation function for the d=1 marginal chain and various values of L. Too few steps results in proposed states being close to the current state and hence a high correlation between samples.

![Autocorrelation Function when L=1](https://github.com/mckimmh/mcmc/blob/main/images/acf_L1_d1.png)

Too many leapfrog steps can result in periodic behaviour, where the chain oscillates from one part of the state space to another, so there is a strong negative correlation between consecutive states.

![Autocorrelation Function when L=16](https://github.com/mckimmh/mcmc/blob/main/images/acf_L16_d1.png)

Just the right number of leapfrog steps (in this case L=7) results in a chain with small autocorrelation, which is highly desirable in MCMC samplers.

![Autocorrelation Function when L=7](https://github.com/mckimmh/mcmc/blob/main/images/acf_L7_d1.png)

The plot below shows a two-dimensional traceplot of the first 20 samples of the (close to) optimally-tuned chain.

![Two-dimensional trace plot of an optimally tune Markov chain generated using HMC](https://github.com/mckimmh/mcmc/blob/main/images/trace_plot2d.png)
