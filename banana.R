# Banana distributrion
b <- 0.03
x1_mean <- 0
x2_mean <- 0
x1_var <- 100
x2_var <- 1 + 100*b + 2*(10^4)*b^2

dbanana <- function(x){
  exp(-0.005*x[1]^2 - 0.5*(x[2] + b*x[1]^2 - 100*b)^2)
}

# Log density
log_dbanana <- function(x){
  -0.005*x[1]^2 - 0.5*(x[2] + b*x[1]^2 - 100*b)^2
}

# Laplace approximation
mu_mean <- c(0, 100*b)
mu_cov <- matrix(c(100, 0, 0, 1), nrow=2)
dmu <- function(x){
  mvtnorm::dmvnorm(x, mu_mean, mu_cov)
}

# Exact density plot
n_cont <- 200
x_seq <- seq(from = -22, to = 22, length.out = n_cont)
y_seq <- seq(from = -12, to = 6, length.out = n_cont)
z <- matrix(0, n_cont, n_cont)
la <- matrix(0, n_cont, n_cont)
for (i in 1:n_cont){
  for (j in 1:n_cont){
    z[i,j] <- dbanana(c(x_seq[i], y_seq[j]))
    la[i,j] <- dmu(c(x_seq[i], y_seq[j]))
  }
}
contour(x_seq, y_seq, z, main='Exact density and Laplace approximation')
contour(x_seq, y_seq, la, add=TRUE, col ='green')

# Perfect samples
n <- 1000000
n_plot <- 1000
x1 <- rnorm(n, sd=10)
x2 <- rnorm(n, sd=1)
y1 <- x1
y2 <- x2 - b*x1^2 + 100*b

# Plot samples
plot(y1[1:n_plot],y2[1:n_plot], pch=20, cex=0.25, xlab='V1', ylab='V2')
contour(x_seq, y_seq, z, col ='gray', add=TRUE)

# Monte Carlo estiamates of moments (check my calculations are correct)
mean(y1)
mean(y2)
mean(y1^2)
mean(y2^2)

# Marginals
plot(density(y1), main='Variable 1')
plot(density(y2), main='Variable 2')

###############################
# Samples from RWM Markov chain
###############################
banana_rwm <- read.table("~/mcmc/banana_rwm_samples.txt",
                             quote="\"", comment.char="")

# Trace plots
if (nrow(banana_rwm) > 1000){
  plot(tail(banana_rwm$V1, 1000), type = 'l', ylab='v1',
       main='Trace plot v1 (RWM)')
  plot(tail(banana_rwm$V2, 1000), type = 'l', ylab='v2',
       main='Trace plot v2 (RWM)')
} else {
  plot(banana_rwm$V1, type = 'l', ylab='v1',
       main='Trace plot v1')
  plot(banana_rwm$V2, type = 'l', ylab='v2',
       main='Trace plot v2')
}

# Auto-correlation (bare in mind the output has already been thinned)
acf(banana_rwm$V1, main='ACF v1 (RWM)')
acf(banana_rwm$V2, main='ACF v2 (RWM)')

# Short 2d trace plot
if ((nrow(banana_rwm) > 50)){
  plot(tail(banana_rwm, 50), type = 'l',
       main='2D trace plot (RWM)')
} else {
  plot(banana_rwm, type = 'l')
}

# Points plot
contour(x_seq, y_seq, z, col = 'gray', main='Points plot (RWM)')
contour(x_seq, y_seq, la, add=TRUE, col ='green')
if ((nrow(banana_rwm) > 500)){
  points(tail(banana_rwm, 500), main='Last samples')
} else {
  points(banana_rwm)
}

# Marginal densities
plot(density(banana_rwm$V1), main='KDE v1 (RWM)')
plot(density(banana_rwm$V2), main='KDE v2 (RWM)')

# First and second moment
colMeans(banana_rwm)
colMeans(banana_rwm^2)

#####################################################
# Markov chain generated with an independence sampler
# based on a Laplace approximation of the target
#####################################################
banana_indep_sampler <- read.table("~/mcmc/banana_indep_sampler.txt",
                                    quote="\"", comment.char="")

# Trace plots
if (nrow(banana_indep_sampler) > 1000){
  plot(tail(banana_indep_sampler$V1, 1000), type = 'l', ylab='v1',
       main='Trace plot v1 (Independence Sampler)')
  plot(tail(banana_indep_sampler$V2, 1000), type = 'l', ylab='v2',
       main='Trace plot v2 (Independence Sampler)')
} else {
  plot(banana_indep_sampler$V1, type = 'l', ylab='v1',
       main='Trace plot v1 (Independence Sampler)')
  plot(banana_indep_sampler$V2, type = 'l', ylab='v2',
       main='Trace plot v2 (Independence Sampler)')
}

# Auto-correlation (bare in mind the output has already been thinned)
acf(banana_indep_sampler$V1, main='ACF v1 (Independence Sampler)')
acf(banana_indep_sampler$V2, main='ACF v2 (Independence Sampler)')

# Short 2d trace plot
if ((nrow(banana_indep_sampler) > 50)){
  plot(tail(banana_indep_sampler, 50), type = 'l',
       main='2D trace plot (Independence Sampler)')
} else {
  plot(banana_indep_sampler, type = 'l')
}

# Points plot
contour(x_seq, y_seq, z, col = 'gray', main='Points plot (Independence Sampler)')
contour(x_seq, y_seq, la, add=TRUE, col ='green')
if ((nrow(banana_indep_sampler) > 500)){
  points(tail(banana_indep_sampler, 500), main='Last samples')
} else {
  points(banana_indep_sampler)
}

# First and second moment
colMeans(banana_indep_sampler)
colMeans(banana_indep_sampler^2)

#################################
# Hamiltonian Monte Carlo Samples
#################################
banana_hmc <- read.table("~/mcmc/banana_hmc_samples.txt",
                                   quote="\"", comment.char="")

# Trace plots
if (nrow(banana_hmc) > 1000){
  plot(tail(banana_hmc$V1, 1000), type = 'l', ylab='v1',
       main='Trace plot v1 (HMC)')
  plot(tail(banana_hmc$V2, 1000), type = 'l', ylab='v2',
       main='Trace plot v2 (HMC)')
} else {
  plot(banana_hmc$V1, type = 'l', ylab='v1',
       main='Trace plot v1 (HMC)')
  plot(banana_hmc$V2, type = 'l', ylab='v2',
       main='Trace plot v2 (HMC)')
}

# Auto-correlation (bare in mind the output has already been thinned)
acf(banana_hmc$V1, main='ACF v1 (HMC)')
acf(banana_hmc$V2, main='ACF v2 (HMC)')

# Short 2d trace plot
if ((nrow(banana_hmc) > 50)){
  plot(tail(banana_hmc, 50), type = 'l',
       main='2D trace plot (HMC)')
} else {
  plot(banana_hmc, type = 'l', main='2D trace plot (HMC)')
}

# Points plot
contour(x_seq, y_seq, z, col = 'gray', main='Points plot (HMC)')
contour(x_seq, y_seq, la, add=TRUE, col ='green')
if ((nrow(banana_hmc) > 500)){
  points(tail(banana_hmc, 500), main='Last samples')
} else {
  points(banana_hmc)
}

# First and second moment
colMeans(banana_hmc)
colMeans(banana_hmc^2)
