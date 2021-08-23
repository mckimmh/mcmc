# Analysis of HMC samples of a 2d Gaussian distribution
library(coda)

# Covariance (and precision) matrix
Sigma <- matrix(c(1.2, 0.4, 0.4, 0.8), nrow=2)

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
