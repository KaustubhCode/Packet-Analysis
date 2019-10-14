
df <- read.csv("/home/kaustubh/Desktop/COL334/Packet-Analysis/inter_arrival_3.csv", header=FALSE)
library("fitdistrplus")
colnames(df) <- "X"
plotdist(df$X, histo=TRUE, demp=TRUE)

plot(df$X)
descdist(df$X)

par(mfrow=c(2,2))
fe <- fitdist(df$X, "exp")
fe
denscomp(list(fe), legendtext=c("exp"))
cdfcomp(list(fe), legendtext=c("exp"))
qqcomp(list(fe), legendtext=c("exp"))

par(mfrow=c(2,2))
fe <- fitdist(df$X, "norm")
fe
denscomp(list(fe), legendtext=c("norm"))
cdfcomp(list(fe), legendtext=c("norm"))
qqcomp(list(fe), legendtext=c("norm"))

par(mfrow=c(2,2))
fe <- fitdist(df$X, "lnorm")
fe
denscomp(list(fe), legendtext=c("lnorm"))
cdfcomp(list(fe), legendtext=c("lnorm"))
qqcomp(list(fe), legendtext=c("lnorm"))