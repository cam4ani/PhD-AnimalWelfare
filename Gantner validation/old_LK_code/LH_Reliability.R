## E2 - Binning validation ###
### Interobserver reliability Video coding ###
### Laura Candelotto March 2020 ###
####################################################################

rm(list=ls())
setwd("//nas-vetsuisse/vetsuisse/Gruppen/VPHI/Welfare/2- Research Projects/Laura C-Klara G/tracking system/binning_validation/1_new videoValidation/results/Reliability")

#### packages ####
library ('tidyverse')
library ("irr")
library ("icr")

#### load the dataset ####
video_Rel <- read_delim("LH_Video_Reliability.csv", ";", escape_double = FALSE, trim_ws = TRUE)
str(video_Rel)

video_Rel$VideoAnalyseDori <- as.factor(video_Rel$VideoAnalyseDori)
video_Rel$VideoAnalyseMelina <- as.factor(video_Rel$VideoAnalyseMelina)
video_Rel$VideoAnalysePatrick <- as.factor(video_Rel$VideoAnalysePatrick)

#################### FULL DATASET ####################################
# including those cases in which the mismatch was due to different interpretations of log files etc

#### kappa ####
# calculate kappa (categorical/nominal data)

# replace NA's with 0 as it is non-agreement between raters
video_Rel$VideoAnalyseDori[is.na(video_Rel$VideoAnalyseDori)] <- 0
video_Rel$VideoAnalyseMelina[is.na(video_Rel$VideoAnalyseMelina)] <- 0
video_Rel$VideoAnalysePatrick[is.na(video_Rel$VideoAnalysePatrick)] <- 0


#kappa for >2 raters: Fleiss' Kappa
kappam.fleiss(video_Rel[,c(7,8,9)], exact=FALSE)
agree(video_Rel[,c(7,8,9)], tolerance=0)


#### binom test ####
# Dori*Melina
binom.test(314, 348, p=0.5) #p<0.001; p=0.9022
# Dori*Patrick
binom.test(310, 348, p=0.5) #p<0.001; p=0.8908
# Melina*Patrick
binom.test(345, 348, p=0.5) #p<0.001; p=0.9914



#### Krippendorff's alpha ####

## reorganize dataset for krippendorff's alpha

# add column with row number
video_Rel <- mutate(video_Rel, ID = seq_len(nrow(video_Rel)))

# create wide dataset with columns being ID
# and rows being raters
ds1 <- gather(video_Rel, Rater, Score, 7:9)
ds2 <- ds1[14:16]
ds2$Score <- as.factor(ds2$Score)

ds_Kripp <- spread(ds2, ID, Score)
ds_Kripp[is.na(ds_Kripp)] <- 2

# calculate Krippendorff's alpha
krippalpha(ds_Kripp, metric = "nominal")

