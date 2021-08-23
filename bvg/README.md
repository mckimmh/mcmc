# Hamiltonian Monte Carlo

## Sampling of a Bivariate Gaussian Distribution

Hamiltonian Monte Carlo is notoriously difficult to tune. One must first choose the leapfrog step-size epsilon. This can be done empirically by choosing a value of epsilon so that the acceptance rate of proposed points remains high. There tends to be a sharp stability limit after which the acceptance rate falls drastically. 

![Stability Limit of the Leapfrog Step-size](https://github.com/mckimmh/mcmc/blob/main/images/leapfrog_step_size.png)
