
df <- read.csv("/home/kaustubh/Desktop/COL334/Packet-Analysis/inter_arrival_3.csv", header=FALSE)
df <- read.csv("W:/IITD Acad/Sem 7/COL334/Ass3/Packet-Analysis/inter_arrival_1.csv", header=FALSE)
library("fitdistrplus")
colnames(df) <- "X"
plotdist(df$X, histo=TRUE, demp=TRUE)

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

###########
D1 (a, b)
Fitting of the distribution ' exp ' by maximum likelihood 
Parameters:
       estimate   Std. Error
rate 0.01947557 0.0004760287

D2 (c, d)
Fitting of the distribution ' norm ' by maximum likelihood 
Parameters:
     estimate Std. Error
mean 51.34637   1.891781
sd   77.19295   1.337690

D3 (e, f)
Fitting of the distribution ' lnorm ' by maximum likelihood 
Parameters:
        estimate Std. Error
meanlog  2.46921 0.06368993
sdlog    2.59883 0.04503555
