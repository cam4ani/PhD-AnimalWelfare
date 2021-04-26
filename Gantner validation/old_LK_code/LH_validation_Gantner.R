## E2 - Stress resilience in Laying hens ###
### Validation Gantner tracking system  barn 4###
### Laura Candelotto Nov 2019 ###
##############################################

rm(list=ls())

#### packages ####

# install.packages("tidyverse") # including ggplot2, dplyr, didyr, readr etc
# install.packages("lubridate")
library ('tidyverse')
library ('irr')

#### SCAN SAMPLING #####################################################################################


#### load the dataset ####
V_Scan <- read_delim("Validation_Scans_v3.csv", ";", escape_double = FALSE, trim_ws = TRUE)
str(V_Scan)
names(V_Scan)

#### reliability ####
kappa2(V_Scan[,c(7,8)], "unweighted") # it is unweighed because it is not worse having a mismatch between e.g. location 1 and 2 in comparison between other locations
agree(V_Scan[,7:8], tolerance=0) # level of agreement is percentage of same data entry; kappa considers the fact that some might be the same due to chance


## for both systems seperately
V_Scan3_5 <- subset(V_Scan, V_Scan$Pen == 3 | V_Scan$Pen == 4 | V_Scan$Pen == 5) # should be 498 entries
V_Scan10_12 <- subset(V_Scan, V_Scan$Pen == 10 | V_Scan$Pen == 11 | V_Scan$Pen == 12) 


kappa2(V_Scan3_5[,c(7,8)], "unweighted") 
agree(V_Scan3_5[,7:8], tolerance=0) 

kappa2(V_Scan10_12[,c(7,8)], "unweighted") 
agree(V_Scan10_12[,7:8], tolerance=0) 




#### Focal observation #########################################################################################


rm(list=ls())
#setwd("G:/VPHI/Welfare/2- Research Projects/Laura C-Klara G/tracking system/validation/focals")
setwd("H:/E2_laying hens/tracking system/validation")

#### load the dataset ####
V_Focal <- read_delim("Validation_Focals_v2.csv", ";", escape_double = FALSE, trim_ws = TRUE)
str(V_Focal)
names(V_Focal)

kappa2(V_Focal[,c(8,9)], "unweighted") 
agree(V_Focal[,8:9], tolerance=0) 

# considering NA's as not matching -> change to "new zone" called x
V_Focal[is.na(V_Focal)]<-7

kappa2(V_Focal[,c(8,9)], "unweighted") 
agree(V_Focal[,8:9], tolerance=0) 

# seperated per tracking system
V_Focal3_5 <- subset(V_Focal, V_Focal$Pen == 3 | V_Focal$Pen == 4 | V_Focal$Pen == 5)
V_Focal10_12 <- subset(V_Focal, V_Focal$Pen == 10 | V_Focal$Pen == 11 | V_Focal$Pen == 12) 


kappa2(V_Focal3_5[,c(8,9)], "unweighted")
agree(V_Focal3_5[,8:9], tolerance=0)

kappa2(V_Focal10_12[,c(8,9)], "unweighted")
agree(V_Focal10_12[,8:9], tolerance=0)

# remove observer mistakes

V_Focal_NM <- subset(V_Focal, V_Focal$ObserverMistake != 1)

V_Focal3_5_NM <- subset(V_Focal_NM, V_Focal_NM$Pen == 3 | V_Focal_NM$Pen == 4 | V_Focal_NM$Pen == 5)
V_Focal10_12_NM <- subset(V_Focal_NM, V_Focal_NM$Pen == 10 | V_Focal_NM$Pen == 11 | V_Focal_NM$Pen == 12) 

kappa2(V_Focal3_5_NM[,c(8,9)], "unweighted")
agree(V_Focal3_5_NM[,8:9], tolerance=0)

kappa2(V_Focal10_12_NM[,c(8,9)], "unweighted")
agree(V_Focal10_12_NM[,8:9], tolerance=0)



#### Time lag #########################################################################################


rm(list=ls())


#### load the dataset ####
V_Mov <- read_delim("Validation_Movement_v2.csv", ";", escape_double = FALSE, trim_ws = TRUE)
str(V_Mov)
names(V_Mov)

#read data
valDat <- read.csv("Validation_TimeLag.csv", header = TRUE, sep = ';')


relData <- valDat[valDat$Pen == 10 |valDat$Pen == 11 |valDat$Pen == 12, ]

calcData <- relData[relData$Track_LT != 'NaN' & relData$Track_LT != '>15' 
                    & relData$Track_RT != '>10' 
                    & relData$Track_RT != 'NaN' & relData$Track_RT != '>15'
                    & relData$Track_MT != 'NaN' & relData$Track_MT != '>15',]


calcData$Mid_Time<-as.POSIXct(calcData$Mid_Time,format="%H:%M:%S")
calcData$Track_MT<-as.POSIXct(calcData$Track_MT,format="%H:%M:%S")
calcData$Left_Time<-as.POSIXct(calcData$Mid_Time,format="%H:%M:%S")
calcData$Track_LT<-as.POSIXct(calcData$Track_MT,format="%H:%M:%S")
calcData$Right_Time<-as.POSIXct(calcData$Mid_Time,format="%H:%M:%S")
calcData$Track_RT<-as.POSIXct(calcData$Track_MT,format="%H:%M:%S")



lag1 <- calcData$Track_MT +1 - calcData$Mid_Time

lag2 <- calcData$Track_LT +1 - calcData$Left_Time

lag3 <- calcData$Track_RT +1 - calcData$Right_Time

Lag <- c(lag1, lag2, lag3)

############################################################################################


median(as.numeric(Lag))
mean(as.numeric(Lag))
max(as.numeric(Lag))
min(as.numeric(Lag))

plot(as.numeric(Lag))








# time difference in seconds between Tracking and Obs
V_Mov_t <- mutate(V_Mov, TimeDiff = difftime(V_Mov$TrackTime, V_Mov$ObsTime,
                                             units = "secs"))

# only instances when both methods were recorded
V_Mov_t <- subset(V_Mov_t, V_Mov_t$TimeDiff != "NA")


### All pens

# max and min value
max(V_Mov_t$TimeDiff)
min(V_Mov_t$TimeDiff) # negative values indicate that the tracking system recorded the zone before the bird was actually moved there
mean(V_Mov_t$TimeDiff)

# percentage of data within the 2s frame (<= 2 / >= -2)
100/nrow(V_Mov_t)*nrow(subset(V_Mov_t, V_Mov_t$TimeDiff<= 2 & V_Mov_t$TimeDiff >= -2))



### seperated per pen type

## floor nest pens
V_Mov_t_Low <- subset(V_Mov_t, V_Mov_t$pen != 2)
V_Mov_t_Low <- subset(V_Mov_t_Low, V_Mov_t_Low$pen != 4)
V_Mov_t_Low <- subset(V_Mov_t_Low, V_Mov_t_Low$pen != 6)
V_Mov_t_Low <- subset(V_Mov_t_Low, V_Mov_t_Low$pen != 8)
V_Mov_t_Low <- subset(V_Mov_t_Low, V_Mov_t_Low$pen != 10)

# max and min value
max(V_Mov_t_Low$TimeDiff) 
min(V_Mov_t_Low$TimeDiff) # negative values indicate that the tracking system recorded the zone before the bird was actually moved there
mean(V_Mov_t_Low$TimeDiff)

# percentage of data within the 2s frame (<= 2 / >= -2)
100/nrow(V_Mov_t_Low)*nrow(subset(V_Mov_t_Low, V_Mov_t_Low$TimeDiff<= 2 & V_Mov_t_Low$TimeDiff >= -2))


## raised nest pens
V_Mov_t_Raised <- subset(V_Mov_t, V_Mov_t$pen != 1)
V_Mov_t_Raised <- subset(V_Mov_t_Raised, V_Mov_t_Raised$pen != 3)
V_Mov_t_Raised <- subset(V_Mov_t_Raised, V_Mov_t_Raised$pen != 5)
V_Mov_t_Raised <- subset(V_Mov_t_Raised, V_Mov_t_Raised$pen != 7)
V_Mov_t_Raised <- subset(V_Mov_t_Raised, V_Mov_t_Raised$pen != 9)

# max and min value
max(V_Mov_t_Raised$TimeDiff) # maybe entry mistake (contact camille) - otherwise 14
min(V_Mov_t_Raised$TimeDiff) # negative values indicate that the tracking system recorded the zone before the bird was actually moved there
mean(V_Mov_t_Raised$TimeDiff)

# percentage of data within the 2s frame (<= 2 / >= -2)
100/nrow(V_Mov_t_Raised)*nrow(subset(V_Mov_t_Raised, V_Mov_t_Raised$TimeDiff<= 2 & V_Mov_t_Raised$TimeDiff >= -2))


# plot
str(V_Mov_t)
V_Mov_t$TimeDiff <- as.numeric(V_Mov_t$TimeDiff)
hist(V_Mov_t$TimeDiff, main="Time difference_full data", breaks=20)



