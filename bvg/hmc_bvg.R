# Analysis of HMC samples of a 2d Gaussian distribution

# Covariance matrix
targ_mean <- c(0,0)
Sigma <- matrix(c(1.0, 0.9, 0.9, 1.0), nrow=2)

# Target density
dtarg <- function(x){
  mvtnorm::dmvnorm(x, targ_mean, Sigma, log=FALSE)
}

# Tuning the leapfrog step-size
epsilon_choice <- read.csv("hmc_bvg_epsilon_avg_accept_rate.txt",
                           sep="")
plot(epsilon_choice, type='l', ylab="Average Acceptance Rate",
     xlab="Leapfrog step-size",
     main="Choice of Leapfrog step-size")

# Tuning the number of leapfrog steps
x <- list()
str1 <- "hmc_bvg_samples_L"
str3 <- ".txt"
for (L in 1:20){
  str2 <- as.character(L)
  str4 <- paste0(str1, str2, str3)
  x[[L]] <- read.table(str4, quote="\"", comment.char="")
}

for (L in 1:20){
  for (d in 1:2){
    str1 <- "ACF, L="
    str2 <- as.character(L)
    str3 <- ", d="
    str4 <- as.character(d)
    acf(x[[L]][,d], main=paste0(str1, str2, str3, str4))
  }
}

# Optimally tuned chain
x <- read.table("hmc_bvg_samples.txt",
                quote="\"", comment.char="")

# Marginal trace plots
for (d in 1:2){
  plot(x[,d], type = 'l')
}

# RMSE of estimates of moments
mo1_est <- colMeans(x)
mo2_est <- colMeans(x^2)

mo1 <- c(0,0)
mo2 <- c(1,1)

rmse1 <- sqrt(mean((mo1 - mo1_est)^2))
rmse2 <- sqrt(mean((mo2 - mo2_est)^2))

# Density estimates
for (i in 1:2){
  plot(density(x[,i]), col = 'blue')
  curve(dnorm, add=TRUE)
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
lines(x[1:20,])


