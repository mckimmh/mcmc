# Analysis of model of Dugongs (sea cows)

x <- c(1.0, 1.5, 1.5, 1.5, 2.5, 4.0, 5.0, 5.0, 7.0,
       8.0, 8.5, 9.0, 9.5, 9.5, 10.0, 12.0, 12.0, 13.0,
       13.0, 14.5, 15.5, 15.5, 16.5, 17.0, 22.5, 29.0, 31.5)

y <- c(1.8, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.35,
       2.47, 2.19, 2.26, 2.40, 2.39, 2.41, 2.50, 2.32, 2.43,
       2.47, 2.56, 2.65, 2.47, 2.64, 2.56, 2.70, 2.72, 2.57)

n <- length(x)
a <- 0.001

# Density of the log target
dlog_targ <- function(theta){
  sum_term <- sum((y - exp(theta[1]) +
                     exp(theta[2])*(exp(theta[3])/(1 + exp(theta[3])))^x)^2)
  sum(theta) - (a + 0.5*n)*log(2*a + sum_term) - 2*log(1 + exp(theta[3]))
}

# Laplace approximation
laplace_info <- optim(c(1, 0, 2), dlog_targ, control=list(fnscale=-1),
                      hessian=TRUE)
mu_mean <- laplace_info$par
mu_cov <- -solve(laplace_info$hessian)

# Load samples
dugong_samples <- read.table("~/rwm/dugong_samples.txt",
                             quote="\"", comment.char="")

# Trace plot
for (i in 1:3){
  plot(dugong_samples[,i], type = 'l', main='Trace Plot',
       ylab=paste('Variable ', i))
}

# Autocorrelation
for (i in 1:3){
  acf(dugong_samples[,i], main=paste('ACF variable', i))
}

# Marginal density estimate in black, Laplace aproximation in green
for (i in 1:3){
  plot(density(dugong_samples[,i]),
       main=paste('Estimate of Marginal Density of variable', i))
  curve(dnorm(x, mean=mu_mean[i], sd=sqrt(mu_cov[i,i])),
        from = min(dugong_samples[,i]), to = max(dugong_samples[,i]),
        col = 'green', add =TRUE)
}

######################
# Independence Sampler
######################
dugong_indep_sampler <- read.table("~/rwm/dugong_indep_sampler.txt",
                                   quote="\"", comment.char="")
# Trace plot
for (i in 1:3){
  plot(dugong_indep_sampler[,i], type = 'l', main='Trace Plot',
       ylab=paste('Variable ', i))
}

# Autocorrelation
for (i in 1:3){
  acf(dugong_indep_sampler[,i], main=paste('ACF variable', i))
}

# Marginal density estimate in black, Laplace aproximation in green
for (i in 1:3){
  plot(density(dugong_indep_sampler[,i]),
       main=paste('Estimate of Marginal Density of variable', i))
  curve(dnorm(x, mean=mu_mean[i], sd=sqrt(mu_cov[i,i])),
        from = min(dugong_indep_sampler[,i]), to = max(dugong_indep_sampler[,i]),
        col = 'green', add =TRUE)
}
