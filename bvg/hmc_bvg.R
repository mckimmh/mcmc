# Analysis of HMC samples of a 2d Gaussian distribution
library(coda)

# Covariance (and precision) matrix
Sigma <- matrix(c(1.0, 0.9, 0.9, 1.0), nrow=2)

# Tuning epsilon
epsilon_choice <- read.csv("hmc_bvg_epsilon_avg_accept_rate.txt",
                           sep="")
plot(epsilon_choice, type='l', ylab="Average Acceptance Rate",
     xlab="Leapfrog step-size",
     main="Choice of Leapfrog step-size")

# Samples
x <- read.table("bvg_hmc_samples.txt",
                quote="\"", comment.char="")
dim(x)
x <- mcmc(x)

summary(x)
traceplot(tail(x, 10000))
autocorr.plot(x)
for (i in 1:2){
  plot(density(x[,i]), col='blue',
       main=paste0('Density estimate X', i))
  curve(dnorm(x, sd=sqrt(Sigma[i,i])), add=TRUE)
}
colMeans(x)
cov(x)
