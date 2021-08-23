# Random Walk Metropolis sampling of a bivariate Gaussian distribution

# Dimension
d <- 2

# Covariance matrix
targ_mean <- c(0,0)
Sigma <- matrix(c(1.0, 0.9, 0.9, 1.0), nrow=2)

# Target density
dtarg <- function(x){
  mvtnorm::dmvnorm(x, targ_mean, Sigma, log=FALSE)
}

# Samples with "small" proposal standard deviation
x_sd_small <- read.table("rwm_samples_sd_small.txt",
                         quote="\"", comment.char="")

# Trace plots
for (i in 1:d){
  str1 <- "Trace Plot X"
  str2 <- as.character(i)
  str3 <- " (small proposal standard deviation)"
  plot(x_sd_small[,i], type = 'l', ylab=paste0('X',i),
       main=paste0(str1, str2, str3))
}

# Samples with acceptance rate roughly 0.234
x_sd_large <- read.table("rwm_samples_sd_large.txt",
                         quote="\"", comment.char="")

for (i in 1:d){
  str1 <- "Trace Plot X"
  str2 <- as.character(i)
  str3 <- " (large proposal standard deviation)"
  plot(x_sd_large[,i], type = 'l', ylab=paste0('X',i),
       main=paste0(str1, str2, str3))
}

# Samples with acceptance rate roughly 0.234
x_sd_tuned <- read.table("rwm_samples_sd_tuned.txt",
                         quote="\"", comment.char="")

for (i in 1:d){
  str1 <- "Trace Plot X"
  str2 <- as.character(i)
  str3 <- " (tuned proposal standard deviation)"
  plot(x_sd_tuned[,i], type = 'l', ylab=paste0('X',i),
       main=paste0(str1, str2, str3))
}

for (i in 1:d){
  str1 <- 'ACF X'
  str2 <- as.character(i)
  str3 <- " (tuned proposal standard deviation)"
  acf(x_sd_tuned[,i], main=paste0(str1, str2, str3))
}

# Contour plots of the density
n <- 100
xs <- seq(from = -3, to = 3, length.out = n)
zs <- matrix(0, nrow=n, ncol=n)
for (i in seq_along(xs)){
  for (j in seq_along(xs)){
    zs[i,j] <- dtarg(c(xs[i], xs[j]))
  }
}
contour(xs, xs, zs, xlab='x1', ylab='x2', nlevels=3, col='gray',
        xlim=c(-2, 2), ylim=c(-2,2))

# 2d traceplot
lines(x_sd_tuned[26:50,], type='b', pch=4)

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
