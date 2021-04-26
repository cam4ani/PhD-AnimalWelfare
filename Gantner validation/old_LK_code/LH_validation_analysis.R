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

setwd("G:/VPHI/Welfare/2- Research Projects/Laura C-Klara G/tracking system/validation/scans")

#### load the dataset ####
V_Scans <- read_delim("Validation_Scans_v4.csv", ";", escape_double = FALSE, trim_ws = TRUE)
str(V_Scans)


# remove all observations of the 22.10.2019 for pens 3-5 (loop 3 not working)

V_Scan <- subset(V_Scans, !((V_Scans$Pen == 3 | 
                   V_Scans$Pen == 4 | 
                       V_Scans$Pen == 5) & V_Scans$Date == "22.10.2019" )) 


#### reliability ####
names(V_Scan)
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
setwd("G:/VPHI/Welfare/2- Research Projects/Laura C-Klara G/tracking system/validation/focals")


#### load the dataset ####
V_Focals <- read_delim("Validation_Focals_v4.csv", ";", escape_double = FALSE, trim_ws = TRUE)
str(V_Focals)


# remove all observations of the 22.10.2019 for pens 3-5 (loop 3 not working)

V_Focal <- subset(V_Focals, !((V_Focals$Pen == 3 | 
                               V_Focals$Pen == 4 | 
                               V_Focals$Pen == 5) & 
                                (V_Focals$Date == "22.10.2019" |
                                   V_Focals$Date == "23.10.2019") ))



# considering NA's as not matching -> change to "new zone" called 7
V_Focal[is.na(V_Focal)]<-7

#### reliability ####
names(V_Focal)

kappa2(V_Focal[,c(8,9)], "unweighted") 
agree(V_Focal[,8:9], tolerance=0) 

# seperated per tracking system
V_Focal3_5 <- subset(V_Focal, V_Focal$Pen == 3 | V_Focal$Pen == 4 | V_Focal$Pen == 5)
V_Focal10_12 <- subset(V_Focal, V_Focal$Pen == 10 | V_Focal$Pen == 11 | V_Focal$Pen == 12) 


kappa2(V_Focal3_5[,c(8,9)], "unweighted")
agree(V_Focal3_5[,8:9], tolerance=0)

kappa2(V_Focal10_12[,c(8,9)], "unweighted")
agree(V_Focal10_12[,8:9], tolerance=0)



#### find mismatches ####

str(V_Focal)

# replace Wintergarten (wg) with a number
V_Focal[V_Focal=="wg"]<-"5"

# change location into numbers
V_Focal$ObsZone <- as.numeric(V_Focal$ObsZone)
V_Focal$TrackZone <- as.numeric(V_Focal$TrackZone)
str(V_Focal)


V_Focal_diff <- mutate(V_Focal, Diff = ObsZone - TrackZone)

V_Focal_mismatch <- subset(V_Focal_diff, V_Focal_diff$Diff != 0)

View(V_Focal_mismatch)
