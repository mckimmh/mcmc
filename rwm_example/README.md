# Random Walk Metropolis

## Implementation



## Example: Sampling from a bivariate Gaussian distribution

We consider using HMC to sample from a bivariate zero-mean Gaussian distribution. Let the variances of both components be 1 and let the covariance between variables be 0.9. Though this is a low-dimensional example, the strong correlation between components makes it an interesting test case. An important tuning parameter for a RWM algorithm with spherical Gaussian proposal distribution is the scale of the proposal. If the standard deviation of the proposal is small, then proposed points are close to the current point, hence tend to have a similar probability density so are often accepted. Though proposed moves are likely to be made, since each is relatively small, the chain mixes slowly. The traceplot below shows an example of a slowly mixing chain, where the standard deviation of each component of the proposal distribution is  0.1.

![](https://github.com/mckimmh/mcmc/blob/main/images/rwm_trace_plot_small_sd.png)

On the other hand, if the standard deviation of each component of the proposal distribution is large, proposed moves tend to be far from the current point, but are likely to be rejected. The traceplot below shows a slowly mixing chain, generated using a proposal distribution with standard deviation 10 for each componet.

![](https://github.com/mckimmh/mcmc/blob/main/images/rwm_trace_plot_large_sd.png)

Optimally the standard deviation of the proposal should be neither too small nor too large. Fortunately, theoretical results exist to help guide the selection of the proposal. The scale should be set so that the average acceptance rate is 0.234. Note that there are caveats to this result, such as applying to high-dimensional distributions with independent components, but it serves as a good rule of thumb for most distributions. The marginal traceplot for a chain with tuned proposal distribution (standard deviation approximately 1.3) is shown below.

![](https://github.com/mckimmh/mcmc/blob/main/images/rwm_trace_plot_tuned_sd.png)
