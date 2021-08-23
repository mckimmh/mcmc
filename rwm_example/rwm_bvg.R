# Random Walk Metropolis sampling of a bivariate Gaussian distribution

# Dimension
d <- 2

# Samples with "small" proposal standard deviation
x_sd_small <- read.table("rwm_samples_sd_small.txt",
                         quote="\"", comment.char="")

# Trace plots
for (i in 1:d){
  plot(x_sd_small[,i], type = 'l')
}

# Samples with acceptance rate roughly 0.234
x_sd_large <- read.table("rwm_samples_sd_large.txt",
                         quote="\"", comment.char="")

for (i in 1:d){
  plot(x_sd_large[,i], type = 'l')
}

# Samples with acceptance rate roughly 0.234
x_sd_tuned <- read.table("rwm_samples_sd_tuned.txt",
                         quote="\"", comment.char="")

for (i in 1:d){
  plot(x_sd_tuned[,i], type = 'l')
}

for (i in 1:d){
  acf(x_sd_tuned[,i])
}

# Thinned and tuned samples
x <- read.table("rwm_samples_sd_tuned_thinned.txt",
                 quote="\"", comment.char="")

# Autocorrelation
for (i in 1:d){
  acf(x[,i])
}

# Check moments
mo1 <- c(0,0)
mo2 <- c(1, 1)
mo1_est <- colMeans(x)
mo2_est <- colMeans(x^2)
rmse1 <- sqrt(mean((mo1 - mo1_est)^2))
rmse2 <- sqrt(mean((mo2 - mo2_est)^2))
rmse1 < 0.01
rmse2 < 0.01

# Density estimates
for (i in 1:d){
  plot(density(x[,i]), col='blue')
  curve(dnorm(x), add=TRUE)
}
