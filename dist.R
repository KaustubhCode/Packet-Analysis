
df <- read.csv("/home/kaustubh/Desktop/COL334/Packet-Analysis/inter_arrival_1.csv", header=FALSE)
df <- read.csv("W:/IITD Acad/Sem 7/COL334/Ass3/Packet-Analysis/inter_arrival_2.csv", header=FALSE)
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

########### Q6
D1 - thresh = 500 ( 3 outliers )
summary statistics
------
min:  0   max:  582.9948 
median:  22.97173 
mean:  52.6001 
estimated sd:  74.8016 
estimated skewness:  2.493592 
estimated kurtosis:  10.64594 

D2 - thresh = 500 ( 1 outlier )
summary statistics
------
min:  0   max:  664.512 
median:  14.82811 
mean:  31.85809 
estimated sd:  45.17995 
estimated skewness:  3.728429 
estimated kurtosis:  29.49672 

D3 - thresh = 600 ( 1 outlier )
summary statistics
------
min:  7.3e-05   max:  750.7615 
median:  20.03094 
mean:  51.34637 
estimated sd:  77.21614 
estimated skewness:  2.937381 
estimated kurtosis:  15.27729 

########## Q7
D1 - thresh = 225 ( 9 outlier )
summary statistics
------
min:  0   max:  293.8761 
median:  0.5502 
mean:  1.132866 
estimated sd:  5.416476 
estimated skewness:  29.4343 
estimated kurtosis:  1109.621 

D2 - thresh = 155 ( 4 outlier )
summary statistics
------
min:  0   max:  188.4669 
median:  0.457643 
mean:  0.9471977 
estimated sd:  3.695835 
estimated skewness:  23.1408 
estimated kurtosis:  708.5954 

D3 - thresh = 400 ( 1 outlier )
summary statistics
------
min:  0   max:  483.5633 
median:  0.553568 
mean:  1.174648 
estimated sd:  6.562533 
estimated skewness:  33.55545 
estimated kurtosis:  1527.231 