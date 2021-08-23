# Hamiltonian Monte Carlo

## Sampling of a Bivariate Gaussian Distribution

Hamiltonian Monte Carlo is notoriously difficult to tune. One must first choose the leapfrog step-size epsilon. This can be done empirically by choosing a value of epsilon so that the acceptance rate of proposed points remains high. There tends to be a sharp stability limit after which the acceptance rate falls drastically. 

![Stability Limit of the Leapfrog Step-size](https://github.com/mckimmh/mcmc/blob/main/images/leapfrog_step_size.png)

Once the a good step-size has been found, choose the number of leapfrog steps (L) so that the autocorrelation of samples is small. Too few steps and the sampler produces a chain with random-walk like behaviour and high autocorrelation

![Autocorrelation Function when L=1](https://github.com/mckimmh/mcmc/blob/main/images/acf_L1_d1.png)

Too many leapfrog steps can generate a Markov chain with periodic behaviour, where the chain oscillates from one part of the state space to the other, so that there is a strong negative correlation between consecutive states.

![Autocorrelation Function when L=16](https://github.com/mckimmh/mcmc/blob/main/images/acf_L16_d1.png)

Just the right number of leapfrog steps, this case L=7, results in a chain with small autocorrelation.

![Autocorrelation Function when L=7](https://github.com/mckimmh/mcmc/blob/main/images/acf_L7_d1.png)
