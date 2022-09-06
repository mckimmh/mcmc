# Random Walk Metropolis

The Random Walk Metropolis (RWM) algorithm [(Metropolis et al., 1953; ](https://aip.scitation.org/doi/abs/10.1063/1.1699114) [Hastings, 1970)](https://academic.oup.com/biomet/article-abstract/57/1/97/284580) is one of the simplest algorithms for Markov Chain Monte Carlo. It generates each element in the Markov chain by first proposing a new state then accepting or rejecting that proposal state according to a certain probability. The accept/reject step ensures that the Markov chain has the correct invariant distribution. The Random Walk Metropolis algorithm uses a Gaussian proposal distribution.

## Implementation

Construct a object of class `RWM` using `RWM mc(log_post, initial_state, burn, thin, n_samples, prop_sd)` where
* `log_post` is an object of class `LogPost` and represents the target log Posterior distribution. This contains the target distribution's dimension, data needed to compute the posterior and a function for evaluating the log probability density
* `initial_state` is the initial state of the Markov chain and has type `arma::vec` 
* `burn` the number of states to discard as a burn-in period
* `thin` the thinning used for the Markov chain
* `n_samples` the number of samples to generate
* `prop_sd` the standard deviation of the spherical Gaussian proposal distribution.
Once the Markov chain has been constructed, you can tune the proposal distribution using `mc.adapt_prop_sd()`, generate the chain using `mc.rwm()` and print samples to the console using `mc.print_chain()` or to an open `std::ofstream` file using `mc.print_chain(file)`.

## Example: Sampling from a bivariate Gaussian distribution

We consider using RWM to sample from a bivariate zero-mean Gaussian distribution. Let the variances of both components be 1 and let the covariance between variables be 0.9. Though this is a low-dimensional example, the strong correlation between components makes it an interesting test case. An important tuning parameter for a RWM algorithm with spherical Gaussian proposal distribution is the scale of the proposal. If the standard deviation of the proposal is small, then proposed points are close to the current point, hence tend to have a similar probability density so are often accepted. Though proposed moves are likely to be made, since each is relatively small, the chain mixes slowly. The traceplot below shows an example of a slowly mixing chain, where the standard deviation of each component of the proposal distribution is  0.1.

![](https://github.com/mckimmh/mcmc/blob/main/images/rwm_trace_plot_small_sd.png)

On the other hand, if the standard deviation of each component of the proposal distribution is large, proposed moves tend to be far from the current point, but are likely to be rejected. The traceplot below shows a slowly mixing chain, generated using a proposal distribution with standard deviation 10 for each componet.

![](https://github.com/mckimmh/mcmc/blob/main/images/rwm_trace_plot_large_sd.png)

Optimally the standard deviation of the proposal should be neither too small nor too large. Fortunately, theoretical results exist to help guide the selection of the proposal [(Roberts et al., 1997)](https://projecteuclid.org/journals/annals-of-applied-probability/volume-7/issue-1/Weak-convergence-and-optimal-scaling-of-random-walk-Metropolis-algorithms/10.1214/aoap/1034625254.full). The scale should be set so that the average acceptance rate is 0.234. Note that there are caveats to this result, such as applying to high-dimensional distributions with independent components, but it serves as a good rule of thumb for most distributions. The marginal traceplot for a chain with tuned proposal distribution (standard deviation approximately 1.3) is shown below.

![](https://github.com/mckimmh/mcmc/blob/main/images/rwm_trace_plot_tuned_sd.png)

A downside of the RWM sampler compared to more advanced methods for MCMC, is that is can generate chains with large autocorrelations. This is illustrated by the figure below, which is a plot of the autocorrelation function for a marginal of the chain generated with optimally tuned proposal distribution.

![](https://github.com/mckimmh/mcmc/blob/main/images/rwm_acf_tuned.png)

Algorithms such as Hamiltonian Monte Carlo are able to generate chains with far small autocorrelations.
