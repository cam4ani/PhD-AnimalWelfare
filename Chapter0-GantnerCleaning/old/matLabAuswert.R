# R Console #
kuhMasse.df <- read.table("R_endtabelle_v3.txt",sep='\t', header=T)

kuhMasse.df<- subset(kuhMasse.df, Outliers == 0)
#kuhMasse.df<- subset(kuhMasse.df, nur_Tape == 0)
#kuhMasse.df<- subset(kuhMasse.df, Outlier_paar == 0)
#kuhMasse.df<- subset(kuhMasse.df, exclude == 0)

## Daten einlesen:
#kuhMasse.df <- read.table ('MATLAB_endtabelle.txt', sep= '\t', header= TRUE)
kuhMasse.df [, 'Kuhnummer'] <- factor (kuhMasse.df [, 'Kuhnummer'])
kuhMasse.df [, 'NrMensch'] <- factor (kuhMasse.df [, 'NrMensch'])
kuhMasse.df [, 'Wiederholung'] <- factor (kuhMasse.df [, 'Wiederholung'])
kuhMasse.df [, 'gerade'] <- as.numeric (kuhMasse.df [, 'Score'] == 0)
summary (kuhMasse.df)

#source ('C:/Users/Yamenah/Desktop/Paper_Manuscript_Matlab/Statistik_Lorenz/plotRel.R')
#source ('plotRel.R')
source ('plotRel_original.R')
library (gmodels)
library (leiv)
library (boot)
library(lme4)
library(pbkrtest)
## Graphik & Statistik
#par (mfrow= c (3, 3), las= 1, bty= 'n', mar= c (4, 4, 0.3, 0.3))
# save as pdf as custom 5.28 x 5.33 inches #
#dev.off()
#Matlab.df <- subset(kuhMasse.df, Method =="MATLAB")
#Tape.df <- subset(kuhMasse.df, Method =="tape")
#summary (Tape.df [Tape.df [, 'Koerpermasse'] == 'IW', 'value_korr_94'])

summary (kuhMasse.df [kuhMasse.df [, 'Koerpermasse'] == 'WH', 'value'])
WH.vc <- plotRel ('WH', 'Withers height', 148)

summary (kuhMasse.df [kuhMasse.df [, 'Koerpermasse'] == 'HH', 'value_korr_94'])
HH.vc <- plotRel ('HH', 'Hip height', 149)

summary (kuhMasse.df [kuhMasse.df [, 'Koerpermasse'] == 'ISH', 'value_korr_94'])
ISH.vc <- plotRel ('ISH', 'Ischium height', 140)

summary (kuhMasse.df [kuhMasse.df [, 'Koerpermasse'] == 'KNH', 'value_korr_94'])
KNH.vc <- plotRel ('KNH', 'Knee height', 94)

summary (kuhMasse.df [kuhMasse.df [, 'Koerpermasse'] == 'SH', 'value_korr_94'])
SH.vc <- plotRel ('SH', 'Shoulder height', 98)

summary (kuhMasse.df [kuhMasse.df [, 'Koerpermasse'] == 'DBL', 'value_korr_94'])
DBL.vc <- plotRel ('DBL', 'Diagonal body length', 175)

summary (kuhMasse.df [kuhMasse.df [, 'Koerpermasse'] == 'BL', 'value_korr_94'])
BL.vc <- plotRel ('BL', 'Body length', 254)

summary (kuhMasse.df [kuhMasse.df [, 'Koerpermasse'] == 'HW', 'value_korr_94'])
HW.vc <- plotRel ('HW', 'Hip width', 60)

summary (kuhMasse.df [kuhMasse.df [, 'Koerpermasse'] == 'IW', 'value_korr_94'])
IW.vc <- plotRel ('IW', 'Ischium width', 30)

par (mfrow= c (1, 1), las= 1, bty= 'n')

## Vergleich Varianzkomponenten
all.vc <- rbind (WH.vc,
                 HH.vc,
                 ISH.vc,
                 KNH.vc,
                 SH.vc,
                 DBL.vc,
                 BL.vc,
                 HW.vc,
                 IW.vc)
names (all.vc)
#### Speichern ####
write.table(all.vc, file="all_VcR_v1.csv", sep = ";", quote=FALSE, col.names=TRUE, row.names=FALSE) # Nur colnames

par(mfrow=c(1,1),las=1,bty='n')
#dat <- read.table("C:/Users/Yamenah/Desktop/Paper_Manuscript_Matlab/Statistik_Lorenz/all_Vc_v1.csv", sep = ";",header=T) # Nur colnames
data <- read.table("all_VcR_v1.csv", sep = ";",header=T) # Nur colnames

#### Beginn Zusatz ####
edit(dat)
dat$method
datTape <- subset (data,data$method == 'tape')
datMatlab <- subset (data,data$method == 'matlab')
messMatlab <-subset(datMatlab,error=="mess")
messTape <-subset(datTape,error=="mess")
obsMatlab <-subset(datMatlab,error=="obs")
obsTape <-subset(datTape,error=="obs")
library("calibrate")
boxplot (var ~ method + error, data=dat, ylab = "variance components [cm]")
messMatlab$var
mess.x <- c(0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2)
mess.y <- c(0.5568547,1.0816306,1.8134402,5.6515736,10.4524778,8.7425124,2.3906315,0.9039519,1.8256486)
points(mess.x,mess.y,cex=.8,pch=16,col="black")
textxy(mess.x, mess.y, messMatlab$Mass)

messTape$var
mess.x1 <- c(1.8,1.85,1.9,1.95,2.0,2.05,2.1,2.15,2.2)
mess.y1 <- c(11.674302,10.052233,13.104502,11.105214,8.009375,46.933265,77.370461,4.616394,24.929444)
points(mess.x1,mess.y1,cex=.8,pch=16,col="black")
textxy(mess.x1, mess.y1, messTape$Mass)

segments(mess.x, mess.y, mess.x1, mess.y1)

obsMatlab$var
obs.x <- c(2.8,2.85,2.9,2.95,3.0,3.05,3.1,3.15,3.2)
obs.y <- c(0.1302195,0.5499819,0.2515346,19.6778410,16.3585528,4.5118880,0.7347106,31.0402893,6.3791879)
points(obs.x,obs.y,cex=.8,pch=16,col="black")
textxy(obs.x, obs.y, obsMatlab$Mass)

obsTape$var
obs.x1 <- c(3.8,3.85,3.9,3.95,4.0,4.05,4.1,4.15,4.2)
obs.y1 <- c(0.2919130,0.2106686,3.3156883,4.9129171,0.4524051,8.3879521,10.6341437,0.6553581,20.6709446)
points(obs.x1,obs.y1,cex=.8,pch=16,col="black")
textxy(obs.x1, obs.y1, obsTape$Mass)

segments(obs.x, obs.y, obs.x1, obs.y1)
#### Ende Zusatz ###

wilcox.test(x, y = NULL,
            alternative = c("two.sided", "less", "greater"),
            mu = 0, paired = FALSE, exact = NULL, correct = TRUE,
            conf.int = FALSE, conf.level = 0.95, ...)

wilcox.test (var ~ method, data, subset= error == 'obs', paired= TRUE)
wilcox.test (var ~ method, data, subset= error == 'mess', paired= TRUE)

wilcox.test (var ~ method, all.vc, subset= error == 'obs', paired= TRUE)
wilcox.test (var ~ method, all.vc, subset= error == 'mess', paired= TRUE)
