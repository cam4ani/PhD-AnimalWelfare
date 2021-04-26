## E2 - Laying hens ###
### Inter and intraobserver reliability health assessments ###
### Laura Candelotto Sept 2020 ###
####################################################################


rm(list=ls())
setwd("//nas-vetsuisse/vetsuisse/Gruppen/VPHI/Welfare/2- Research Projects/Laura C-Klara G/Health assessment/Reliability")

#### packages ####

# install.packages("tidyverse") # including ggplot2, dplyr, didyr, readr etc

library ('tidyverse')
library ('irr')

#### load the dataset ####
# HA_rel <- read_delim("Interobserver reliability health measurements_d1.csv", ";", escape_double = FALSE, trim_ws = TRUE)
HA_rel <- read_delim("HARel_2020_07_27.csv", ";", escape_double = FALSE, trim_ws = TRUE)
str(HA_rel)

HA_rel$pen <- as.factor(HA_rel$pen)
HA_rel$Round <- as.factor(HA_rel$Round)
HA_rel$Rater <- as.factor(HA_rel$Rater)
HA_rel$bare_head <- as.factor(HA_rel$bare_head)
HA_rel$barePatch_head <- as.factor((HA_rel$barePatch_head))
names(HA_rel)

str(HA_rel)

# calculate sum and average for feet and feather coverage
HA_rel$sum_podo <- HA_rel$l_podo + HA_rel$r_podo
HA_rel$mean_podo <- rowMeans(HA_rel[,c('l_podo', 'r_podo')])

HA_rel$sum_bumble <- HA_rel$l_bumble + HA_rel$r_bumble
HA_rel$mean_bumble <- rowMeans(HA_rel[,c('l_bumble', 'r_bumble')])

HA_rel$sum_injure <- HA_rel$l_injure + HA_rel$r_injure
HA_rel$mean_injure <- rowMeans(HA_rel[,c('l_injure', 'r_injure')])

HA_rel$sum_social <- HA_rel$cloaca + HA_rel$tail
HA_rel$mean_social <- rowMeans(HA_rel[,c('cloaca', 'tail')])

HA_rel$sum_wear <- HA_rel$neck + HA_rel$wings + HA_rel$breast
HA_rel$mean_wear <- rowMeans(HA_rel[,c('wings', 'neck', 'breast')])

HA_rel$sum_featherCoverage <- HA_rel$neck + HA_rel$wings + HA_rel$tail + HA_rel$cloaca + HA_rel$breast
HA_rel$mean_featherCoverage <- rowMeans(HA_rel[,c('neck', 'wings', 'tail', 'cloaca', 'breast')])


# subset for each observer
HA_K <- subset(HA_rel, HA_rel$Rater == "Klara")
HA_L <- subset(HA_rel, HA_rel$Rater == "Laura")





#### INTRA-OBSERVER RELIABILITY ####
#________________________________________________________________________________________________

#### Klara ####

#### continuous variables ####
# (interclass correlation)

#HA_A_intra <- subset(HA_A, HA_A$Pen != "1") # nur f?r das dataset _d1
#HA_A_1 <- subset(HA_A_intra, HA_A_intra$round == "1")
#HA_A_2 <- subset(HA_A_intra, HA_A_intra$round == "2")

HA_K_1 <- subset(HA_K, HA_K$Round == "1")
HA_K_2 <- subset(HA_K, HA_K$Round == "2")

names(HA_K)


## feather coverage continuous

# neck
a= data.frame(HA_K_1$neck)
b= data.frame(HA_K_2$neck)
x= data.frame (a,b)
icc(x, model="twoway", type="agreement")
#faster writing
icc(data.frame(HA_K_1$neck, HA_K_2$neck), model="twoway", type="agreement")

# wing
icc(data.frame(HA_K_1$wings, HA_K_2$wings), model="twoway", type="agreement")

# tail
icc(data.frame(HA_K_1$tail, HA_K_2$tail), model="twoway", type="agreement")

# cloaca
icc(data.frame(HA_K_1$cloaca, HA_K_2$cloaca), model="twoway", type="agreement")

# breast
icc(data.frame(HA_K_1$breast, HA_K_2$breast), model="twoway", type="agreement")

# Feather coverage_social
icc(data.frame(HA_K_1$sum_social, HA_K_2$sum_social), model="twoway", type="agreement")
icc(data.frame(HA_K_1$mean_social, HA_K_2$mean_social), model="twoway", type="agreement")

# Feather coverage_wear
icc(data.frame(HA_K_1$sum_wear, HA_K_2$sum_wear), model="twoway", type="agreement")
icc(data.frame(HA_K_1$mean_wear, HA_K_2$mean_wear), model="twoway", type="agreement")

# Feather coverage total
icc(data.frame(HA_K_1$sum_featherCoverage, HA_K_2$sum_featherCoverage), model="twoway", type="agreement")
icc(data.frame(HA_K_1$mean_featherCoverage, HA_K_2$mean_featherCoverage), model="twoway", type="agreement")
# ICC is always the same for sum and mean
# makes sense as average is just sum/x and thus it should have the same reliability

## Wounds
icc(data.frame(HA_K_1$wounds, HA_K_2$wounds), model="twoway", type="agreement")


## Feet continuous

# pododermatitis R
icc(data.frame(HA_K_1$r_podo, HA_K_2$r_podo), model="twoway", type="agreement")

# pododermatitis L
icc(data.frame(HA_K_1$l_podo, HA_K_2$l_podo), model="twoway", type="agreement")

#pododermatitis sum
icc(data.frame(HA_K_1$sum_podo, HA_K_2$sum_podo), model="twoway", type="agreement")



# Bumblefoot R
icc(data.frame(HA_K_1$r_bumble, HA_K_2$r_bumble), model="twoway", type="agreement")

# Bumblefoot L
icc(data.frame(HA_K_1$l_bumble, HA_K_2$l_bumble), model="twoway", type="agreement")
# ICC of 0 seems very unlikely
hist(HA_K_1$l_bumble, xlim=c(-2, 2)) #all zeros
hist(HA_K_2$l_bumble)

# Bumblefoot sum
icc(data.frame(HA_K_1$sum_bumble, HA_K_2$sum_bumble), model="twoway", type="agreement")



# Injuries R
icc(data.frame(HA_K_1$r_injure, HA_K_2$r_injure), model="twoway", type="agreement")
#again ICC of 0
hist(HA_K_1$r_injure, xlim=c(-2, 2)) #all zeros
hist(HA_K_2$r_injure)

# Injuries L
icc(data.frame(HA_K_1$l_injure, HA_K_2$l_injure), model="twoway", type="agreement")
#again very awkward ICC
hist(HA_K_1$l_injure)
hist(HA_K_2$l_injure)
#calculate the difference between entries (diff of zero shows perfect match)
x <- data.frame(HA_K_1$l_injure, HA_K_2$l_injure)
x$diff_l_injure <- x$HA_K_2.l_injure - x$HA_K_1.l_injure
hist(x$diff_l_injure, xlim=c(-10, 10)) #... hmmm sieht nicht nach ICC = 0.1 aus ... sehr komisch....

# leg injuries sum
icc(data.frame(HA_K_1$sum_injure, HA_K_2$sum_injure), model="twoway", type="agreement")
x <- data.frame(HA_K_1$sum_injure, HA_K_2$sum_injure)
x$diff_sum_injure <- x$HA_K_2.sum_injure - x$HA_K_1.sum_injure
hist(x$diff_sum_injure, xlim=c(-10, 10)) #sieht nicht so schlimm aus wie der ICC




#### categorical variables ####
# (kappa -> for categorical data)

# bare head
a= data.frame(HA_K_1$bare_head)
b= data.frame(HA_K_2$bare_head)
x= data.frame (a,b)
kappa2(x[,c(1,2)], "unweighted") 
# change all "halbe glatze"  in the comments (and all bare_head=1) into barePatch_head = 1
a= data.frame(HA_K_1$barePatch_head)
b= data.frame(HA_K_2$barePatch_head)
x= data.frame (a,b)
kappa2(x[,c(1,2)], "unweighted") 
# problem with binom data?
x <- data.frame(HA_K_1$barePatch_head, HA_K_2$barePatch_head)
x$diff_barePatch_head <- x$HA_K_2.barePatch_head - x$HA_K_1.barePatch_head
hist(x$diff_barePatch_head, xlim=c(-2, 2)) #... ok, gseht eigentlech nid so schlimm us... -> 77% stimmen ?berein


# missing toe
a= data.frame(HA_K_1$missing_toe)
b= data.frame(HA_K_2$missing_toe)
x= data.frame (a,b)
kappa2(x[,c(1,2)], "unweighted") 
# kappa = 1; mal gucken ob das wirklich 100% ?bereinstimmt
x <- data.frame(HA_K_1$missing_toe, HA_K_2$missing_toe) #passt






#### Laura ####

#### continuous variables ####
# (interclass correlation)

HA_L_1 <- subset(HA_L, HA_L$Round == "1")
HA_L_2 <- subset(HA_L, HA_L$Round == "2")

names(HA_L)

## feather coverage

# neck
icc(data.frame(HA_L_1$neck, HA_L_2$neck), model="twoway", type="agreement")

# wing
icc(data.frame(HA_L_1$wings, HA_L_2$wings), model="twoway", type="agreement")

# tail
icc(data.frame(HA_L_1$tail, HA_L_2$tail), model="twoway", type="agreement")

# cloaca
icc(data.frame(HA_L_1$cloaca, HA_L_2$cloaca), model="twoway", type="agreement")

# breast
icc(data.frame(HA_L_1$breast, HA_L_2$breast), model="twoway", type="agreement")

# Feather coverage_social
icc(data.frame(HA_L_1$sum_social, HA_L_2$sum_social), model="twoway", type="agreement")

# Feather coverage_wear
icc(data.frame(HA_L_1$sum_wear, HA_L_2$sum_wear), model="twoway", type="agreement")

# Feather coverage total
icc(data.frame(HA_L_1$sum_featherCoverage, HA_L_2$sum_featherCoverage), model="twoway", type="agreement")


## Wounds
icc(data.frame(HA_L_1$wounds, HA_L_2$wounds), model="twoway", type="agreement")



## Feet continuous

# pododermatitis R
icc(data.frame(HA_L_1$r_podo, HA_L_2$r_podo), model="twoway", type="agreement")

# pododermatitis L
icc(data.frame(HA_L_1$l_podo, HA_L_2$l_podo), model="twoway", type="agreement")

# pododermatitis sum
icc(data.frame(HA_L_1$sum_podo, HA_L_2$sum_podo), model="twoway", type="agreement")



# Bumblefoot R
icc(data.frame(HA_L_1$r_bumble, HA_L_2$r_bumble), model="twoway", type="agreement")

# Bumblefoot L
icc(data.frame(HA_L_1$l_bumble, HA_L_2$l_bumble), model="twoway", type="agreement")
# ICC is NaN
hist(HA_L_1$l_bumble, xlim=c(-2, 2)) #all zeros
hist(HA_L_2$l_bumble, xlim=c(-2, 2)) #all zeros
# technically it is a 100% match

# bumblefoot sum
icc(data.frame(HA_L_1$sum_bumble, HA_L_2$sum_bumble), model="twoway", type="agreement")



# Injuries R
icc(data.frame(HA_L_1$r_injure, HA_L_2$r_injure), model="twoway", type="agreement")
#ICC of 0
hist(HA_L_1$r_injure, xlim=c(-2, 2)) #all zeros
hist(HA_L_2$r_injure)


# Injuries L
icc(data.frame(HA_L_1$l_injure, HA_L_2$l_injure), model="twoway", type="agreement")

# injuries sum
icc(data.frame(HA_L_1$sum_injure, HA_L_2$sum_injure), model="twoway", type="agreement")

x <- data.frame(HA_L_1$sum_injure, HA_L_2$sum_injure)
x$diff_sum_injure <- x$HA_L_2.sum_injure - x$HA_L_1.sum_injure
hist(x$diff_sum_injure, xlim=c(-10, 10)) #sieht nicht so schlimm aus wie der ICC





#### categorical variables ####
# (kappa -> for categorical data)

# bare head
a= data.frame(HA_L_1$bare_head)
b= data.frame(HA_L_2$bare_head)
x= data.frame (a,b)
kappa2(x[,c(1,2)], "unweighted") 
# change all "halbe glatze"  in the comments (and all bare_head=1) into barePatch_head = 1
a= data.frame(HA_L_1$barePatch_head)
b= data.frame(HA_L_2$barePatch_head)
x= data.frame (a,b)
kappa2(x[,c(1,2)], "unweighted") 
# problem with binom data?
x <- data.frame(HA_L_1$barePatch_head, HA_L_2$barePatch_head)
x$diff_barePatch_head <- x$HA_L_2.barePatch_head - x$HA_L_1.barePatch_head
hist(x$diff_barePatch_head, xlim=c(-2, 2)) #... ok, gseht eigentlech nid so schlimm us... -> 77% stimmen ?berein




# missing toe
a= data.frame(HA_L_1$missing_toe)
b= data.frame(HA_L_2$missing_toe)
x= data.frame (a,b)
kappa2(x[,c(1,2)], "unweighted") 





#### INTER-OBSERVER RELIABILITY ####
#________________________________________________________________________________________________


rm(list=ls())

# HA_rel <- read_delim("Interobserver reliability health measurements_d1.csv", ";", escape_double = FALSE, trim_ws = TRUE)
HA_rel <- read_delim("HARel_2020_07_27.csv", ";", escape_double = FALSE, trim_ws = TRUE)
HA_rel$pen <- as.factor(HA_rel$pen)
HA_rel$Round <- as.factor(HA_rel$Round)

HA_rel$sum_podo <- HA_rel$l_podo + HA_rel$r_podo
HA_rel$sum_bumble <- HA_rel$l_bumble + HA_rel$r_bumble
HA_rel$sum_injure <- HA_rel$l_injure + HA_rel$r_injure
HA_rel$sum_social <- HA_rel$cloaca + HA_rel$tail
HA_rel$sum_wear <- HA_rel$neck + HA_rel$wings + HA_rel$breast
HA_rel$sum_featherCoverage <- HA_rel$neck + HA_rel$wings + HA_rel$tail + HA_rel$cloaca + HA_rel$breast

HA_K <- subset(HA_rel, HA_rel$Rater == "Klara")
HA_L <- subset(HA_rel, HA_rel$Rater == "Laura")

names(HA_rel)

# neck
icc(data.frame(HA_K$neck, HA_L$neck), model="twoway", type="agreement")

# Wing
icc(data.frame(HA_K$wings, HA_L$wings), model="twoway", type="agreement")

# Tail
icc(data.frame(HA_K$tail, HA_L$tail), model="twoway", type="agreement")

# Cloaca
icc(data.frame(HA_K$cloaca, HA_L$cloaca), model="twoway", type="agreement")

# Breast
icc(data.frame(HA_K$breast, HA_L$breast), model="twoway", type="agreement")

# feather coverage_social
icc(data.frame(HA_K$sum_social, HA_L$sum_social), model="twoway", type="agreement")

# feather coverage_wear
icc(data.frame(HA_K$sum_wear, HA_L$sum_wear), model="twoway", type="agreement")

# feather coverage total
icc(data.frame(HA_K$sum_featherCoverage, HA_L$sum_featherCoverage), model="twoway", type="agreement")



# Wounds
icc(data.frame(HA_K$wounds, HA_L$wounds), model="twoway", type="agreement")


## Feet

# Pododermatitis R
icc(data.frame(HA_K$r_podo, HA_L$r_podo), model="twoway", type="agreement")

# Pododermatitis L
icc(data.frame(HA_K$l_podo, HA_L$l_podo), model="twoway", type="agreement")

# Pododermatitis Sum
icc(data.frame(HA_K$sum_podo, HA_L$sum_podo), model="twoway", type="agreement")



# Bumblefoot R
icc(data.frame(HA_K$r_bumble, HA_L$l_bumble), model="twoway", type="agreement")
# hahaha ICC is 9e-17, this can't be true
hist(HA_L$r_bumble) 
hist(HA_K$r_bumble)
x <- data.frame(HA_L$r_bumble, HA_K$r_bumble)
x$diff_r_bumble <- x$HA_K.r_bumble - x$HA_L.r_bumble
hist(x$diff_r_bumble) #great majority is identical...
# mal alles + 1 rechnen falls die Nullen das Problem sind
HA_K$r_bumble <- HA_K$r_bumble + 1
icc(data.frame(HA_K$r_bumble, HA_L$l_bumble), model="twoway", type="agreement")
#nope

# Bumblefoot L
icc(data.frame(HA_K$l_bumble, HA_L$l_bumble), model="twoway", type="agreement")
#nope

# Bumblefoot Sum
icc(data.frame(HA_K$sum_bumble, HA_L$sum_bumble), model="twoway", type="agreement")



# Injuries R
icc(data.frame(HA_K$r_injure, HA_L$r_injure), model="twoway", type="agreement")

# Injuries L
icc(data.frame(HA_K$l_injure, HA_L$l_injure), model="twoway", type="agreement")
#awkward
hist(HA_K$l_injure)
hist(HA_L$l_injure)
x <- data.frame(HA_L$l_injure, HA_K$l_injure)
x$diff_l_injure <- x$HA_K.l_injure - x$HA_L.l_injure
hist(x$diff_l_injure) #great majority is identical, but few big outliers

# injuries Sum
icc(data.frame(HA_K$sum_injure, HA_L$sum_injure), model="twoway", type="agreement")
hist(HA_K$sum_injure)
hist(HA_L$sum_injure)
x <- data.frame(HA_L$sum_injure, HA_K$sum_injure)
x$diff_sum_injure <- x$HA_K.sum_injure - x$HA_L.sum_injure
hist(x$diff_sum_injure) 



#### categorical variables ####
# (kappa -> for categorical data)

# bare head
a= data.frame(HA_K$bare_head)
b= data.frame(HA_L$bare_head)
x= data.frame (a,b)
kappa2(x[,c(1,2)], "unweighted") 

# barePatch head
a= data.frame(HA_K$barePatch_head)
b= data.frame(HA_L$barePatch_head)
x= data.frame (a,b)
kappa2(x[,c(1,2)], "unweighted")
x <- data.frame(HA_L$barePatch_head, HA_K$barePatch_head)
x$diff_barePatch_head <- x$HA_K.barePatch_head - x$HA_L.barePatch_head
x$diff_barePatch_head <- as.numeric(x$diff_barePatch_head)
hist(x$diff_barePatch_head) 
table(x$diff_barePatch_head)


# try krippendorffs alpha
library(irr)
library(reshape2)

HeadPatch <- HA_rel[,c(2:5, 26)]
HeadPatch$ID <- paste(HeadPatch$pen, HeadPatch$backpack)
HeadPatch <- arrange(HeadPatch, desc(Rater))
#HeadPatch <- HeadPatch %>% mutate(No = row_number())
HeadPatch$No <- rep(1:52, 2)

datawide <- dcast(HeadPatch, No + ID ~ Rater, value.var= "barePatch_head")
datawide$No <- NULL
datawide$Subject <- NULL
datawide <- as.matrix(t(datawide))


kripp.alpha(datawide, method = "nominal")


# next try with Kappa
a= data.frame(HA_K$barePatch_head)
b= data.frame(HA_L$barePatch_head)
x= data.frame (a,b)
x$HA_K.barePatch_head <- as.factor(x$HA_K.barePatch_head)
x$HA_L.barePatch_head <- as.factor(x$HA_L.barePatch_head)
kappa2(x,  "equal", sort.levels = FALSE) #ok, looks already better


# missing toe
a= data.frame(HA_K$missing_toe)
b= data.frame(HA_L$missing_toe)
x= data.frame (a,b)
kappa2(x[,c(1,2)], "unweighted") 



####################################### DISTRIBUTION PLOTS ############################################################################

rm(list=ls())

# HA_rel <- read_delim("Interobserver reliability health measurements_d1.csv", ";", escape_double = FALSE, trim_ws = TRUE)
HA_rel <- read_delim("HARel_2020_07_27.csv", ";", escape_double = FALSE, trim_ws = TRUE)
HA_rel$pen <- as.factor(HA_rel$pen)
HA_rel$Round <- as.factor(HA_rel$Round)
HA_rel <- subset(HA_rel, HA_rel$pen != "NA")

HA_rel$sum_podo <- HA_rel$l_podo + HA_rel$r_podo
HA_rel$sum_bumble <- HA_rel$l_bumble + HA_rel$r_bumble
HA_rel$sum_injure <- HA_rel$l_injure + HA_rel$r_injure
HA_rel$sum_social <- HA_rel$cloaca + HA_rel$tail
HA_rel$sum_wear <- HA_rel$neck + HA_rel$wings + HA_rel$breast
HA_rel$sum_featherCoverage <- HA_rel$neck + HA_rel$wings + HA_rel$tail + HA_rel$cloaca + HA_rel$breast

names(HA_rel)


# head
ggplot(HA_rel, aes(x=bare_head, fill=Round)) +
  geom_histogram(position="dodge") +
  facet_grid(~Rater)+theme_bw()
ggplot(HA_rel, aes(x=barePatch_head, fill=Round)) +
  geom_histogram(position="dodge") +
  facet_grid(~Rater)+theme_bw()

# neck
ggplot(HA_rel, aes(x=neck, fill=Round)) +
  geom_histogram(position="dodge") +
  facet_grid(~Rater)+theme_bw() +
  ggtitle("Feather coverage neck")


# cloaca
ggplot(HA_rel, aes(x=cloaca, fill=Round)) +
  geom_histogram(position="dodge") +
  facet_grid(~Rater)+theme_bw() +
  ggtitle("Feather coverage cloaca")


# Pododermatitis R
ggplot(HA_rel, aes(x=r_podo, fill=Round)) +
  geom_histogram(position="dodge") +
  facet_grid(~Rater)+theme_bw() +
  ggtitle("Pododermatitis right")


# Pododermatitis sum
ggplot(HA_rel, aes(x=sum_podo, fill=Round)) +
  geom_histogram(position="dodge") +
  facet_grid(~Rater)+theme_bw() +
  ggtitle("Pododermatitis sum")

# Bumblefoot sum
ggplot(HA_rel, aes(x=sum_bumble, fill=Round)) +
  geom_histogram(position="dodge") +
  facet_grid(~Rater)+theme_bw() +
  ggtitle("bumblefoot sum")

# Leg injuries sum
ggplot(HA_rel, aes(x=sum_injure, fill=Round)) +
  geom_histogram(position="dodge") +
  facet_grid(~Rater)+theme_bw() +
  ggtitle("leg injuries sum")

# Feather coverage sum
ggplot(HA_rel, aes(x=sum_featherCoverage, fill=Round)) +
  geom_histogram(position="dodge") +
  facet_grid(~Rater)+theme_bw() +
  ggtitle("Feather coverage sum")


########################################
#### EINSCHUB YG ####
# also ich habe die Formel mal näher angeschaut und, wenn natürlich kaum Varianz sowohl innerhalb 
# als auch zwischen gegeben ist, weil alle Werte praktisch identisch (in diesem Falle = 0) dann ist quasi 
# die vereinfachte Version der Formel:
# Wahre_Varianz / (Wahre_Varianz + Fehler_Varianz) und Wahre_Varianz irgendwo sehr nahe bei 0
# und Fehler_Varianz riiiiiesig, weil kaum ABweichung und wenn dann gleich OUtlierabweichung
# dies ergibt "kleine Zahl" dividiert durch "riesige Zahl", ergo ICC Wert sehr niedrig.

# Ich habe dann überlegt, ob allenfalls mittels Bootstrapping pro RUnde zufällig die Sample neu 
# gemischelt und gezogen werden könne, ob man damit die OUtlier Problematik aus dem Weg gehen könne:

# bootstrapping von ICC mit Outliers
origdat <- data.frame(HA_K$sum_injure, HA_L$sum_injure)

icc_values <- NULL

for(i in 1:500){
  bootdat <- sample_n(origdat,dim(origdat)[1],replace = TRUE)
  iccres <- icc(bootdat, model="twoway", type="agreement")
  icc_values <- c(icc_values,iccres$value)
}

mean(icc_values)
quantile(icc_values,probs = c(0.025,0.975))
range(icc_values)


str(iccres)

# es hat leider das Problem nicht gelöst, aber evtl. liegt es ja daran, dass sum injuries nicht 
# kontinulierliche Daten sind sondern count data?
# wenn dem so wäre, würde ich kein ICC rechnen sondern wieder aufs andere zurückgreifen und 
# kappa mit match zu non-match Vergleich. Denn wenn bei 52 Datenpunkte, gerade mal max 8 von 0 abweichen
# hast du eine totale zero inflation:
plot(HA_K$sum_injure)
plot(HA_L$sum_injure)

# Versuch mal zu schauen, ob dies hilft.
# also bei count data kappa zu nehmen und bei allen kontinuierlichen Daten die tiefe ICC haben den obigen bootstrap anzuwenden.

# alternativ könnte auch ein glmer mit familiy poisson helfen:
mod <-  glmer(y~(1|rater), family=poisson) 
summary(mod)                                        

## Varianz des Random Terms, also Residuals ==> Berechnung vom ICC:
v.RE  <- summary(mod)$varcor$id[1]  
v.res <- var(resid(mod))            
ICC   <- v.RE/(v.RE+v.res)

summary(mod)
summary()