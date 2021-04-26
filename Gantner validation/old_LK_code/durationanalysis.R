library(splitstackshape)
library(ggplot2)
library(reshape)

setwd("//nas-vetsuisse/vetsuisse/Gruppen/VPHI/Welfare/2- Research Projects/Laura C-Klara G/tracking system/video analysis")

data1 <- read.csv("flick1.csv", header = TRUE, sep = ';')
data2 <- read.csv("flick2.csv", header = TRUE, sep = ';')
data3 <- read.csv("flick3.csv", header = TRUE, sep = ';')


data1$POSIX =  as.POSIXct(data1$Time, format = '%H:%M:%S')
data2$POSIX =  as.POSIXct(data2$Time, format = '%H:%M:%S')
data3$POSIX =  as.POSIXct(data3$Time, format = '%H:%M:%S')


data1$dur =  c(data1$POSIX[2:length(data1$POSIX)]-data1$POSIX[1:length(data1$POSIX)-1], 0)
data2$dur =  c(data2$POSIX[2:length(data2$POSIX)]-data2$POSIX[1:length(data2$POSIX)-1], 0)
data3$dur =  c(data3$POSIX[2:length(data3$POSIX)]-data3$POSIX[1:length(data3$POSIX)-1], 0)

d1 = vector(mode = 'integer', length= 3)
d2 = vector(mode = 'integer', length= 4)
d3 = vector(mode = 'integer', length= 5)

j = 0
for (i in levels(data1$Transition.logged)){
  j = j+ 1
  d1[j] = sum(data1$dur[data1$Transition.logged == i])
}
j = 0
for (i in levels(data2$Transition.Logged)){
  j = j+ 1
  d2[j] = sum(data2$dur[data2$Transition.Logged == i])
}
j = 0
for (i in levels(data3$Transition.logged)){
  j = j+ 1
  d3[j] = sum(data3$dur[data3$Transition.logged == i])
}