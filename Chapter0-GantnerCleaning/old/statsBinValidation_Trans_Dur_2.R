### VOR ANWENDUNG VON GGBIPLOT UND SCREEPLOT DIE BEIDEN ENTSPRECHENDEN 
### R SCRIPTS IM SELBEN FOLDeR WIE DIESES ZUERST LAUFEN LASSEN!! ####
# -> ich glaube das loest man indem man source('your_script.R') hinzufuegt, hab das mal gemacht (KG)

#install.packages("Rcpp")
library("tidyverse")
#install.packages('lpSolve')
#library("irr")
#library("rlang")
#install.packages("devtools")
#library("devtools")
#options(download.file.method = "wininet")
#install_github("vqv/ggbiplot", force=T)
#library("ggbiplot")

#clear workspace
rm(list = ls())

#set working directory
#setwd("G:/VPHI/Welfare/2- Research Projects/Laura C-Klara G/tracking system/binning_validation)
setwd("//nas-vetsuisse/vetsuisse/Gruppen/VPHI/Welfare/2- Research Projects/Laura C-Klara G/tracking system/binning_validation")
#in case the university internet breaks again and we need to put files onto the desktop:
#setwd("C:/Users/candelotto/Desktop/for Laura")

#load the dataset
ds <- read_delim("Trans_Dur_Comparison_withPen.csv", ";", escape_double = FALSE, trim_ws = TRUE)
#View(ds)

ds$Bin <- as.factor(ds$Bin)
ds$Pen <- as.factor(ds$Pen)


# create a dataset with all differences
#################################################################################


# creating goldstandard-video dataframe:

goldStandard = data.frame(totTrans = ds$totTransVideo[ds$Bin==1], 
                          dur1 = ds$Dur1Video[ds$Bin==1],dur2 = ds$Dur2Video[ds$Bin==1],
                          dur3 = ds$Dur3Video[ds$Bin==1],dur4 = ds$Dur4Video[ds$Bin==1],
                          dur5 = ds$Dur5Video[ds$Bin==1])
#and bin data frames:
bin1 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==1], 
                  dur1 = ds$Dur1LogBin[ds$Bin==1],dur2 = ds$Dur2LogBin[ds$Bin==1],
                  dur3 = ds$Dur3LogBin[ds$Bin==1],dur4 = ds$Dur4LogBin[ds$Bin==1],
                  dur5 = ds$Dur5LogBin[ds$Bin==1])

bin2 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==2], 
                  dur1 = ds$Dur1LogBin[ds$Bin==2],dur2 = ds$Dur2LogBin[ds$Bin==2],
                  dur3 = ds$Dur3LogBin[ds$Bin==2],dur4 = ds$Dur4LogBin[ds$Bin==2],
                  dur5 = ds$Dur5LogBin[ds$Bin==2])

bin3 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==3], 
                  dur1 = ds$Dur1LogBin[ds$Bin==3],dur2 = ds$Dur2LogBin[ds$Bin==3],
                  dur3 = ds$Dur3LogBin[ds$Bin==3],dur4 = ds$Dur4LogBin[ds$Bin==3],
                  dur5 = ds$Dur5LogBin[ds$Bin==3])

bin4 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==4], 
                  dur1 = ds$Dur1LogBin[ds$Bin==4],dur2 = ds$Dur2LogBin[ds$Bin==4],
                  dur3 = ds$Dur3LogBin[ds$Bin==4],dur4 = ds$Dur4LogBin[ds$Bin==4],
                  dur5 = ds$Dur5LogBin[ds$Bin==4])

bin5 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==5], 
                  dur1 = ds$Dur1LogBin[ds$Bin==5],dur2 = ds$Dur2LogBin[ds$Bin==5],
                  dur3 = ds$Dur3LogBin[ds$Bin==5],dur4 = ds$Dur4LogBin[ds$Bin==5],
                  dur5 = ds$Dur5LogBin[ds$Bin==5])


# A-B diffmatrix

comp1 = as.matrix(goldStandard) - as.matrix(bin1)
comp2 = as.matrix(goldStandard) - as.matrix(bin2)
comp3 = as.matrix(goldStandard) - as.matrix(bin3)
comp4 = as.matrix(goldStandard) - as.matrix(bin4)
comp5 = as.matrix(goldStandard) - as.matrix(bin5)



#dataset with all differences

comp.all = data.frame(bin = c(rep(1,54),rep(2,54),rep(3,54),rep(4,54),rep(5,54)),
                      tag = ds$Tag, 
                      pen = ds$Pen)
comp.all = cbind(comp.all, rbind(comp1, comp2, comp3, comp4, comp5))






# PLOTS ########################################################################################

library(ggplot2)

ggplot(comp.all, aes(factor(bin), totTrans)) + 
  geom_violin()+ geom_jitter(height = 0, width = 0.2)+ 
xlab('Bin') + ylab('total Tranisions')+
  ggtitle('Total number of transitions')

ggplot(comp.all, aes(totTrans, colour = factor(bin))) + 
  geom_density()+
  xlab('Difference to gold standard') + ylab('Density')+
  ggtitle('Total number of transitions')+
  geom_vline(xintercept = 0, linetype = 'dashed')


# durations

# all durations
names(comp.all)
helper <- subset(comp.all[,c(1:3, 5:9)])
names(helper)
comp.all.Dur <- gather(helper, "Zone", "DurationDiff", c(4:8))
comp.all.Dur$Zone <- as.factor(gsub('[a-zA-Z]', '', comp.all.Dur$Zone))

ggplot(comp.all.Dur, aes(DurationDiff, colour = factor(bin))) + 
  geom_density()+
  xlab('Difference to gold standard') + ylab('Density')+
  ggtitle('Duration irrespective of zone')+
  geom_vline(xintercept = 0, linetype = 'dashed')



# per zone
ggplot(comp.all, aes(factor(bin), dur1)) + coord_cartesian(ylim = c(-100, 100))+
  geom_violin()+ geom_jitter(height = 0, width = 0.2) +
  xlab('Bin Size') + ylab('Difference to gold Standard')+
  ggtitle('Duration in zone 1')

ggplot(comp.all, aes(factor(bin), dur2)) + 
  geom_violin()+ geom_jitter(height = 0, width = 0.2) + coord_cartesian(ylim = c(-1000, 1000))+
  xlab('Bin Size') + ylab('Difference to gold Standard')+
  ggtitle('Duration in zone 2')

ggplot(comp.all, aes(factor(bin), dur3)) + 
  geom_violin()+ geom_jitter(height = 0, width = 0.2) + coord_cartesian(ylim = c(-1000, 500))+
  xlab('Bin Size') + ylab('Difference to gold Standard')+
  ggtitle('Duration in zone 3')

ggplot(comp.all, aes(factor(bin), dur4)) + 
  geom_violin()+ geom_jitter(height = 0, width = 0.2) + coord_cartesian(ylim = c(-200, 200))+
  xlab('Bin Size') + ylab('Difference to gold Standard')+
  ggtitle('Duration in zone 4')

ggplot(comp.all, aes(factor(bin), dur5)) + 
  geom_violin()+ geom_jitter(height = 0, width = 0.2) + coord_cartesian(ylim = c(-250, 250))+
  xlab('Bin Size') + ylab('Difference to gold Standard')+
  ggtitle('Duration in zone 5')


ggplot(comp.all, aes(dur3, colour = factor(bin))) + 
  geom_density()+
  xlab('Difference to gold standard') + ylab('Density')+
  ggtitle('Total duration in zone 3')+
  geom_vline(xintercept = 0, linetype = 'dashed')

ggplot(comp.all, aes(dur1, colour = factor(bin))) + 
  geom_density()+
  xlab('Difference to gold standard') + ylab('Density')+
  ggtitle('Total duration in zone 1')+
  geom_vline(xintercept = 0, linetype = 'dashed')
ggplot(comp.all, aes(dur2, colour = factor(bin))) + 
  geom_density()+
  xlab('Difference to gold standard') + ylab('Density')+
  ggtitle('Total duration in zone 2')+
  geom_vline(xintercept = 0, linetype = 'dashed')
ggplot(comp.all, aes(dur4, colour = factor(bin))) + 
  geom_density()+
  xlab('Difference to gold standard') + ylab('Density')+
  ggtitle('Total duration in zone 4')+
  geom_vline(xintercept = 0, linetype = 'dashed')
ggplot(comp.all, aes(dur5, colour = factor(bin))) + 
  geom_density()+
  xlab('Difference to gold standard') + ylab('Density')+
  ggtitle('Total duration in zone 5')+
  geom_vline(xintercept = 0, linetype = 'dashed')






# CORRELATIONS #################################################################
rm(list=ls())
ds <- read_delim("Trans_Dur_Comparison_withPen.csv", ";", escape_double = FALSE, trim_ws = TRUE)
ds$Bin <- as.factor(ds$Bin)
ds$Pen <- as.factor(ds$Pen)

# subsets per bin
ds1 <- subset(ds, ds$Bin==1)
ds2 <- subset(ds,ds$Bin==2)
ds3 <- subset(ds,ds$Bin==3)
ds4 <- subset(ds,ds$Bin==4)
ds5 <- subset(ds,ds$Bin==5)

# transitions video-log (independent of bin size)
cor(ds1$totTransLog, ds1$totTransVideo, use="complete.obs", method="spearman") #0.5616
plot(ds1$totTransLog, ds1$totTransVideo)

# durations video-log (independent of bin size)
names(ds)
dsNeu <- subset(ds[,c(1:5, 10, 14, 18, 22, 26)])
dsNeu2 <- subset(ds[,c(1:5, 11, 15, 19, 23, 27)])
dsDur1 <- gather(dsNeu, "Zone", "DurationLog", c(6:10))
dsDur1$Zone <- as.integer(gsub('[a-zA-Z]', '', dsDur1$Zone))
dsDur2 <- gather(dsNeu2, "Zone", "DurationVideo", c(6:10))
dsDur2$Zone <- as.integer(gsub('[a-zA-Z]', '', dsDur2$Zone))

dsDur <- merge(dsDur1, dsDur2, by = c("Bin", "Date", "Tag", "BirdID", "Pen", "Zone"))


cor(dsDur$DurationLog, dsDur$DurationVideo) #0.972


## correlation matrix
library("Hmisc")


### needed functions
myspread <- function(df, key, value) {
  # quote key
  keyq <- rlang::enquo(key)
  # break value vector into quotes
  valueq <- rlang::enquo(value)
  s <- rlang::quos(!!valueq)
  df %>% gather(variable, value, !!!s) %>%
    unite(temp, !!keyq, variable) %>%
    spread(temp, value)
}
# reorganize correlation matrix
# cormat : matrix of the correlation coefficients
# pmat : matrix of the correlation p-values
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}


## transitions
dsTrans <- ds[, 1:8]
names(dsTrans)

dsTrans_spread <- dsTrans %>% myspread(Bin, totTransLogBin)

cmTrans <- rcorr(as.matrix(dsTrans_spread[, 5:11]))
cmTrans$r #correlations
cmTrans$P #p values

CorrTrans <- flattenCorrMatrix(cmTrans$r, cmTrans$P)
CorrTrans_Video <- subset(CorrTrans, CorrTrans$row=="totTransVideo" | CorrTrans$column=="totTransVideo")

# visualize correlation matrix
library("corrplot")
corrplot(cmTrans$r, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)


## durations
names(ds)
dsNeu_logBin <- subset(ds[,c(1:5, 12, 16, 20, 24, 28)])
dsNeu_video <- subset(ds[,c(1:5, 11, 15, 19, 23, 27)])
dsNeu_log <- subset(ds[,c(1:5, 10, 14, 18, 22, 26)])
dsDur1 <- gather(dsNeu_logBin, "Zone", "DurationLogBin", c(6:10))
dsDur1$Zone <- as.integer(gsub('[a-zA-Z]', '', dsDur1$Zone))
dsDur2 <- gather(dsNeu_video, "Zone", "DurationVideo", c(6:10))
dsDur2$Zone <- as.integer(gsub('[a-zA-Z]', '', dsDur2$Zone))
dsDur3 <- gather(dsNeu_log, "Zone", "DurationLog", c(6:10))
dsDur3$Zone <- as.integer(gsub('[a-zA-Z]', '', dsDur2$Zone))

dsDur <- merge(dsDur1, dsDur2, by = c("Bin", "Date", "Tag", "BirdID", "Pen", "Zone"))
#doesn't work for 3 datasets
# merge 3 datasets by rownames
rn <- rownames(dsDur1)
l <- list(df1, df2, df3, df4)
dat <- l[[1]]
for(i in 2:length(l)) {
  dat <- merge(dat, l[[i]],  by= "row.names", all.x= F, all.y= F) [,-1]
  rownames(dat) <- rn
}

dsDur_spread <- dsDur %>% myspread(Bin, DurationLogBin)

library("Hmisc")
cmDur <- rcorr(as.matrix(dsDur_spread[, 6:11]))
CorrDur <- flattenCorrMatrix(cmDur$r, cmDur$P)
CorrDur_Video <- subset(CorrDur, CorrDur$row=="DurationVideo" | CorrDur$column=="DurationVideo")

# visualize correlation matrix
library("corrplot")
corrplot(cmDur$r, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

# ABSOLUTE DIFFERENCES ######################################################################################## 
rm(list=ls())
ds <- read_delim("Trans_Dur_Comparison_withPen.csv", ";", escape_double = FALSE, trim_ws = TRUE)
ds$Bin <- as.factor(ds$Bin)
ds$Pen <- as.factor(ds$Pen)

names(ds)

# creating goldstandard-video dataframe:

goldStandard = data.frame(totTrans_GB = ds$totTransVideo[ds$Bin==1], 
                          dur1_GB = ds$Dur1Video[ds$Bin==1],
                          dur2_GB = ds$Dur2Video[ds$Bin==1],
                          dur3_GB = ds$Dur3Video[ds$Bin==1],
                          dur4_GB = ds$Dur4Video[ds$Bin==1],
                          dur5_GB = ds$Dur5Video[ds$Bin==1])

# creating original log file dataframe
originalLog = data.frame(totTrans_OB  = ds$totTransLog[ds$Bin==1], 
                         dur1_OB = ds$Dur1Log[ds$Bin==1],
                         dur2_OB = ds$Dur2Log[ds$Bin==1],
                         dur3_OB = ds$Dur3Log[ds$Bin==1],
                         dur4_OB = ds$Dur4Log[ds$Bin==1],
                         dur5_OB = ds$Dur5Log[ds$Bin==1])

#and bin data frames:
bin1 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==1], 
                  dur1 = ds$Dur1LogBin[ds$Bin==1],dur2 = ds$Dur2LogBin[ds$Bin==1],
                  dur3 = ds$Dur3LogBin[ds$Bin==1],dur4 = ds$Dur4LogBin[ds$Bin==1],
                  dur5 = ds$Dur5LogBin[ds$Bin==1])

bin2 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==2], 
                  dur1 = ds$Dur1LogBin[ds$Bin==2],dur2 = ds$Dur2LogBin[ds$Bin==2],
                  dur3 = ds$Dur3LogBin[ds$Bin==2],dur4 = ds$Dur4LogBin[ds$Bin==2],
                  dur5 = ds$Dur5LogBin[ds$Bin==2])

bin3 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==3], 
                  dur1 = ds$Dur1LogBin[ds$Bin==3],dur2 = ds$Dur2LogBin[ds$Bin==3],
                  dur3 = ds$Dur3LogBin[ds$Bin==3],dur4 = ds$Dur4LogBin[ds$Bin==3],
                  dur5 = ds$Dur5LogBin[ds$Bin==3])

bin4 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==4], 
                  dur1 = ds$Dur1LogBin[ds$Bin==4],dur2 = ds$Dur2LogBin[ds$Bin==4],
                  dur3 = ds$Dur3LogBin[ds$Bin==4],dur4 = ds$Dur4LogBin[ds$Bin==4],
                  dur5 = ds$Dur5LogBin[ds$Bin==4])

bin5 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==5], 
                  dur1 = ds$Dur1LogBin[ds$Bin==5],dur2 = ds$Dur2LogBin[ds$Bin==5],
                  dur3 = ds$Dur3LogBin[ds$Bin==5],dur4 = ds$Dur4LogBin[ds$Bin==5],
                  dur5 = ds$Dur5LogBin[ds$Bin==5])


# A-B diffmatrix

comp_GB_1 = as.matrix(goldStandard) - as.matrix(bin1)
comp_GB_2 = as.matrix(goldStandard) - as.matrix(bin2)
comp_GB_3 = as.matrix(goldStandard) - as.matrix(bin3)
comp_GB_4 = as.matrix(goldStandard) - as.matrix(bin4)
comp_GB_5 = as.matrix(goldStandard) - as.matrix(bin5)

comp_OB_1 = as.matrix(originalLog) - as.matrix(bin1)
comp_OB_2 = as.matrix(originalLog) - as.matrix(bin2)
comp_OB_3 = as.matrix(originalLog) - as.matrix(bin3)
comp_OB_4 = as.matrix(originalLog) - as.matrix(bin4)
comp_OB_5 = as.matrix(originalLog) - as.matrix(bin5)

#dataset with all differences

comp.all.prep = data.frame(bin = c(rep(1,54),rep(2,54),rep(3,54),rep(4,54),rep(5,54)),
                      tag = ds$Tag, 
                      pen = ds$Pen,
                      date = ds$Date)
# checking if all rows are where they are supposed to be:
#comp.all.check.GB = cbind(comp.all.prep, rbind(comp_GB_1, comp_GB_2, comp_GB_3, comp_GB_4, comp_GB_5))
#comp.all.check.OB = cbind(comp.all.prep, rbind(comp_OB_1, comp_OB_2, comp_OB_3, comp_OB_4, comp_OB_5))
#comp.all.check = cbind(comp.all.prep, comp.all.check.GB, comp.all.check.OB)

# create actual file:
diff.all.GB = rbind(comp_GB_1, comp_GB_2, comp_GB_3, comp_GB_4, comp_GB_5)
diff.all.OB = rbind(comp_OB_1, comp_OB_2, comp_OB_3, comp_OB_4, comp_OB_5)
diff.all = cbind(comp.all.prep, diff.all.GB, diff.all.OB)

diff.all$bin <- as.factor(diff.all$bin)
summary(abs(diff.all[5:16]))

# summary for transitions
tapply(diff.all$totTrans_GB, diff.all$bin, summary)
tapply(diff.all$totTrans_OB, diff.all$bin, summary)

# Summary for durations
# reorganize the data
# GB
Neu <- subset(diff.all[,c(1:4, 6:10)])
diff.all.Dur.GB <- gather(Neu, "Zone", "DurationDiff_GB", c(5:9))
diff.all.Dur.GB$Zone <- as.factor(gsub('[a-zA-Z]', '', diff.all.Dur.GB$Zone))
#OB
Neu1 <- subset(diff.all[,c(1:4, 12:16)])
diff.all.Dur.OB <- gather(Neu1, "Zone", "DurationDiff_OB", c(5:9))
diff.all.Dur.OB$Zone <- as.factor(gsub('[a-zA-Z]', '', diff.all.Dur.OB$Zone))
# merge into one dataset
diff.all.Dur <- merge(diff.all.Dur.GB, diff.all.Dur.OB, by=c("bin", "tag", "pen", "date", "Zone"), all.x = TRUE)
diff.all.Dur$bin <- as.factor(diff.all.Dur$bin)
#summary
tapply(diff.all.Dur$DurationDiff_GB, diff.all.Dur$bin, summary)
tapply(diff.all.Dur$DurationDiff_OB, diff.all.Dur$bin, summary)
# summary per bin per zone
tapply(diff.all.Dur$DurationDiff_GB, diff.all.Dur$bin:diff.all.Dur$Zone, summary)
tapply(diff.all.Dur$DurationDiff_OB, diff.all.Dur$bin:diff.all.Dur$Zone, summary)


# absolute difference between original and video
# (no bins and therefore less observations)
names(goldStandard)
names(originalLog)

comp_GO = as.matrix(goldStandard) - as.matrix(originalLog)
comp.all.prep = data.frame(tag = ds$Tag[ds$Bin==1], 
                           pen = ds$Pen[ds$Bin==1],
                           date = ds$Date[ds$Bin==1])
diff.GO = cbind(comp.all.prep, comp_GO)

#transitions
summary(diff.GO$totTrans_GB)

# durations
names(diff.GO)
Neu2 <- subset(diff.GO[,c(1:3, 5:9)])
diff.all.Dur.GO <- gather(Neu2, "Zone", "DurationDiff_GO", c(4:8))
diff.all.Dur.GO$Zone <- as.factor(gsub('[a-zA-Z]', '', diff.all.Dur.GO$Zone))
summary(diff.all.Dur.GO$DurationDiff_GO)
tapply(diff.all.Dur.GO$DurationDiff_GO, diff.all.Dur.GO$Zone, summary)




# MODEL #############################################################
#comp.all -> Differences: bin, totTrans, dur1, etc.


#### Transitions ####



qqnorm(comp.all$totTrans)
qqline(comp.all$totTrans)

comp.all$bin = as.factor(comp.all$bin)

library(lme4)
modTrans = lmer(totTrans ~ bin + (1|pen/tag), comp.all)
library(multcomp)
summary(glht(modTrans, linfct= mcp(bin= 'Tukey')), test = adjusted('bonferroni'))

library(DHARMa)
resid.df<- simulateResiduals(modTrans, 1000)
plotSimulatedResiduals(resid.df)

# histograms of the data
comp.all.bin1 <- subset(comp.all, comp.all$bin=="1")
hist(comp.all.bin1$totTrans, breaks=20, ylim=c(0,20), xlim=c(-5, 15), col="gray")
comp.all.bin2 <- subset(comp.all, comp.all$bin=="2")
hist(comp.all.bin2$totTrans, breaks=20, ylim=c(0,20), xlim=c(-5, 15), col="gray")
comp.all.bin3 <- subset(comp.all, comp.all$bin=="3")
hist(comp.all.bin3$totTrans, breaks=20, ylim=c(0,20), xlim=c(-5, 15), col="gray")


# diagnostics problematic
#binomial approach (match vs non match)
comp.all$BinomTrans <- comp.all$totTrans
comp.all$BinomTrans[comp.all$BinomTrans != 0] <- 1
comp.all$bin <- as.factor(comp.all$bin)

library(lme4)
modTrans <- glmer(BinomTrans ~ bin 
                + (1|pen/tag), data = comp.all,
                family = binomial)
# singular fit je nach computer 
library(DHARMa)
resid.df<- simulateResiduals(modTrans, 1000)
plotSimulatedResiduals(resid.df)


# reduce bin (creating a NULL Model)
modNULL <- update(modTrans, .~.-bin)
anova(modTrans, modNULL)

# significant (p=0.017)
summary(modTrans)
anova(modTrans)
library(multcomp)
summary(glht(modTrans, linfct= mcp(bin= 'Tukey')), test = adjusted('bonferroni'))




#### Durations ####

#library(reshape2)
#test = melt(comp.all, id.vars = c('tag', 'bin','pen'), zone= c('dur1', 'dur2', 'dur3', 'dur4', 'dur5'))
#mod2 = lmer(value ~ bin*variable + (1|pen/tag), test)

Neu <- subset(comp.all[,c(1:3, 5:9)])
comp.all.Dur <- gather(Neu, "Zone", "DurationDiff", c(4:8))

comp.all.Dur$bin <- as.factor(comp.all.Dur$bin)
comp.all.Dur$Zone <- as.factor(comp.all.Dur$Zone)


# match vs no match in durations (zero vs not zero in difference)
comp.all.Dur$Binom <- comp.all.Dur$DurationDiff
comp.all.Dur$Binom[comp.all.Dur$Binom != 0] <- 1

library(lme4)
modDur <- glmer(Binom ~ bin + Zone + bin*Zone 
                + (1|pen/tag), data = comp.all.Dur,
                family = binomial)


# reduce bin*Zone
modDur1 <- update(modDur, .~.-bin*Zone)
anova(modDur, modDur1)

# create interaction column
comp.all.Dur$BZ <- interaction(comp.all.Dur$bin, comp.all.Dur$Zone)
modDur <- glmer(Binom ~ BZ + (1|pen/tag), data = comp.all.Dur, family = binomial)

library(multcomp)
summary(glht(modDur, linfct= mcp(BZ= 'Tukey')), test = adjusted('bonferroni'))

# for the bin comparison, the zone is actually not relevant,
# as it should work for all zones
modDur <- glmer(Binom ~ bin 
                + (1|pen/tag), data = comp.all.Dur,
                family = binomial)

# reduce bin (creating a NULL Model)
modDur1 <- update(modDur, .~.-bin)
anova(modDur, modDur1)

# no significant difference between bin sizes


# 2. Step: To what extent do we have a mismatch: for this we only focus on the subset of data
# with mismatched rows only and apply a negative binomial mixed model
# since we are not interested in direction of mismatch (which method has higher values or lower)
# we can use absolute values and apply a negative binomial model
# However, this step is a bit tricky to argue because negative binomial asks for integer data, which 
# normally come from count data. Here we take the difference between times. Since seconds are the smallest
# unit the data should be integer and can to some extent be considered as count and this fact would
# also explain the weird structure found in the BA plots. This distribution is actually typically for count data

mismatch.df <- subset(comp.all.Dur, Binom==1)
mismatch.df$DurationDiff <- abs(mismatch.df$DurationDiff)
hist(mismatch.df$DurationDiff)
qqnorm(mismatch.df$DurationDiff)
qqline(mismatch.df$DurationDiff)

modDur <- glmer.nb(DurationDiff ~ bin + Zone + bin*Zone 
                   + (1|pen/tag), data = mismatch.df)

library(DHARMa)
resid.df<- simulateResiduals(modDur, 1000)
plotSimulatedResiduals(resid.df)


# reduce bin*Zone
modDur1 <- update(modDur, .~.-bin*Zone)
anova(modDur, modDur1)

# create interaction column
comp.all.Dur$BZ <- interaction(comp.all.Dur$Zone, comp.all.Dur$bin)
modDur <- glmer.nb(abs(DurationDiff) ~ BZ + (1|pen/tag), data = mismatch.df)

library(multcomp)
summary(glht(modDur, linfct= mcp(BZ= 'Tukey')), test = adjusted('bonferroni'))

### Bin only
# 1. Step
modDur <- glmer(Binom ~ bin + (1|pen/tag), data = comp.all.Dur,
                family = binomial)
# reduce bin*Zone
modDur1 <- update(modDur, .~.-bin)
anova(modDur, modDur1)

library(multcomp)
summary(glht(modDur, linfct= mcp(bin= 'Tukey')), test = adjusted('bonferroni'))

# game over: no difference in bin size for duration

# 2.Step
mismatch.df <- subset(comp.all.Dur, Binom==1)

modDur2 <- glmer.nb(abs(DurationDiff) ~ bin 
                   + (1|pen/tag), data = mismatch.df)

library(DHARMa)
resid.df<- simulateResiduals(modDur2, 1000)
plotSimulatedResiduals(resid.df)

modDur3 <- update(modDur2, .~.-bin)
anova(modDur2, modDur3)

library(multcomp)
summary(glht(modDur, linfct= mcp(bin= 'Tukey')), test = adjusted('bonferroni'))

### GAMMA ### if not wanting to go for considering differences between durations 
# as number of seconds and as such as count data

modDur <- glmer((abs(DurationDiff)+1) ~ bin + (1|pen/tag), data = comp.all.Dur,
                family = Gamma(link=identity))
summary(glht(modDur, linfct= mcp(bin= 'Tukey')), test = adjusted('bonferroni'))

# bin 1 und 2 machen keinen signifikanten Unterschied.







################## alternative approach #######################################
###########################################################################
rm(list = ls())

# load the data
setwd("//nas-vetsuisse/vetsuisse/Gruppen/VPHI/Welfare/2- Research Projects/Laura C-Klara G/tracking system/binning_validation")
library(tidyverse)
ds <- read_delim("Trans_Dur_Comparison1.csv", ";", escape_double = FALSE, trim_ws = TRUE)
ds$Bin <- as.factor(ds$Bin)
ds$Pen <- as.factor(ds$Pen)

# reorganize the dataset 
Neu <- subset(ds[,c(1:5, 12, 16, 20, 24, 28)])
ds.Dur.log <- gather(Neu, "Zone", "Duration", c(6:10))
ds.Dur.log$Zone <- as.integer(gsub("[a-zA-Z]", "", ds.Dur.log$Zone))

Neu <- subset(ds[,c(1:5, 11, 15, 19, 23, 27)])
ds.Dur.Video <- gather(Neu, "Zone", "Duration", c(6:10))
ds.Dur.Video$Zone <- as.integer(gsub("[a-zA-Z]", "", ds.Dur.Video$Zone))

ds.Dur.Video$Method <- paste("Video")
ds.Dur.log$Method <- paste("BinLog")
ds.Dur <- rbind(ds.Dur.Video, ds.Dur.log)

hist(ds.Dur$Duration)
str(ds.Dur)


ds.Dur <- as.data.frame(ds.Dur)


# load the plotRel Function
source("plotRel_Function.R")

# pro bin model mit Zone als ZV

summary (ds.Dur.Video [ds.Dur.Video [, "Bin"] == "1", "Duration"]) 
summary (ds.Dur.Video [ds.Dur.Video [, "Bin"] == "2", "Duration"]) 

#summary (ds.Dur.Video [ds.Dur.Video [, "Zone"] == "Dur1Video", "Duration"]) # dies pro Bin Size und Methode getrennt vornehmen



bin1.vc <- plotRel ("1", "bin1", 720) 




## Vergleich Varianzkomponenten
all.vc <- rbind (bin1.vc#,  # oder Zone per bin ebene
                 #HH.vc,
                 #ISH.vc,
                 #KNH.vc,
                 #SH.vc,
)
names (all.vc)








#####################################################################################
# former tries
#####################################################################################

# change it to absolute values and try with poisson distribution
comp.all.Dur$DurationDiff <- abs(comp.all.Dur$DurationDiff)
hist(comp.all.Dur$DurationDiff)


####
#standarized version
Neu <- subset(comp.all.S[,c(1:3, 5:9)])
comp.all.Dur.S <- gather(Neu, "Zone", "DurationDiff", c(4:8))

comp.all.Dur.S$bin <- as.factor(comp.all.Dur.S$bin)
comp.all.Dur.S$Zone <- as.factor(comp.all.Dur.S$Zone)

comp.all.Dur.S$DurationDiff <- comp.all.Dur.S$DurationDiff + 3000
comp.all.Dur.S$DurationDiff <- abs(comp.all.Dur.S$DurationDiff)
hist(comp.all.Dur.S$DurationDiff)
qqnorm(comp.all.Dur.S$DurationDiff)
qqline(comp.all.Dur.S$DurationDiff)


library(lme4)
modDur <- glmer(DurationDiff ~ bin + Zone + bin*Zone 
                + (1|pen/tag), data = comp.all.Dur,
                family = poisson)

library(DHARMa)
resid.df <- simulateResiduals(modDur, 1000)
plotSimulatedResiduals(resid.df)

#Overdispersion #
overdisp_fun <- function(model) {
  rdf <- df.residual(model)
  rp <- residuals(model,type="pearson")
  Pearson.chisq <- sum(rp^2)
  prat <- Pearson.chisq/rdf
  pval <- pchisq(Pearson.chisq, df=rdf, lower.tail=F)
  c(chisq=Pearson.chisq,ratio=prat, rdf=rdf, p=pval)
}
overdisp_fun(Agg)



library(MASS)

comp.all.Dur$DurationDiff <- comp.all.Dur$DurationDiff + 1
comp.all.Dur$DurationDiff <- comp.all.Dur$DurationDiff - 1
hist(comp.all.Dur$DurationDiff)

library(boxcoxmix)
comp.all.Dur$DurationDiffBox <- boxcox(comp.all.Dur$DurationDiff ~1)
str(comp.all.Dur$DurationDiff)


Box = boxcox(comp.all.Dur$DurationDiff ~ 1,              # Transform Turbidity as a single vector
             lambda = seq(-6,6,0.1)                      # Try values -6 to 6 by 0.1
)

Cox = data.frame(Box$x, Box$y)
Cox2 = Cox[with(Cox, order(-Cox$Box.y)),] # Order the new data frame by decreasing y

Cox2[1,]                                  # Display the lambda with the greatest log likelihood

lambda = Cox2[1, "Box.x"]                 # Extract that lambda

Diff_box = (comp.all.Dur$DurationDiff ^ lambda - 1)/lambda   # Transform the original data

hist(Diff_box)
qqnorm(Diff_box)
qqline(Diff_box)


modDur <- glmer(boxcox(DurationDiff) ~ bin + Zone + bin*Zone 
                + (1|pen/tag), data = comp.all.Dur,
                family = Gamma(link=log))
#, control=glmerControl(optimizer="bobyqa"))

library(DHARMa)
resid.df <- simulateResiduals(modDur, 1000)
plotSimulatedResiduals(resid.df)




modDur <- lmer(DurationDiff ~ bin + Zone + bin*Zone + (1|pen/tag), comp.all.Dur)

library(DHARMa)
resid.df <- simulateResiduals(modDur, 1000)
plotSimulatedResiduals(resid.df)

hist(comp.all.Dur$DurationDiff)
qqnorm(comp.all.Dur$DurationDiff)
qqline(comp.all.Dur$DurationDiff)


modDur <- glmer(DurationDiff ~ bin + Zone + bin*Zone 
                + (1|pen/tag), data = comp.all.Dur,
                family = Gamma(link = log))



# subsets per bin
ds1 <- subset(ds, ds$Bin==1)
ds2 <- subset(ds,ds$Bin==2)
ds3 <- subset(ds,ds$Bin==3)
ds4 <- subset(ds,ds$Bin==4)
ds5 <- subset(ds,ds$Bin==5)


#### PCA #####
source('ggbiplot.R')
source('ggscreeplot.R')

# all bin sizes
pca.labels <- ds[,c(1:8)]
final.pca <- ds[,c(5:8)]
Trans.pca <- princomp(final.pca, cor= T, center = TRUE,scale. = TRUE)
ggbiplot(Trans.pca, ellipse=T, obs.scale=1, var.scale=1,labels = pca.labels$BirdID, groups=pca.labels$Bin)

#each bin seperately
pca.labels1 <- ds1[,c(1:8)]
final.pca1 <- ds1[,c(5:8)]
Trans.pca1 <- princomp(final.pca1, cor= T, center = TRUE,scale. = TRUE)
summary(Trans.pca1)
Trans.pca1$loadings

pca.labels2 <- ds2[,c(1:8)]
final.pca2 <- ds2[,c(5:8)]
Trans.pca2 <- princomp(final.pca2, cor= T, center = TRUE,scale. = TRUE)
summary(Trans.pca2)
Trans.pca2$loadings

pca.labels3 <- ds3[,c(1:8)]
final.pca3 <- ds3[,c(5:8)]
Trans.pca3 <- princomp(final.pca3, cor= T, center = TRUE,scale. = TRUE)
summary(Trans.pca3)
Trans.pca3$loadings

# plot
ggbiplot(Trans.pca1, ellipse=T, obs.scale=1, var.scale=1,labels = pca.labels1$BirdID)
ggbiplot(Trans.pca2, ellipse=T, obs.scale=1, var.scale=1,labels = pca.labels2$BirdID)
ggbiplot(Trans.pca3, ellipse=T, obs.scale=1, var.scale=1,labels = pca.labels3$BirdID)




## Correlations ##

## transitions per bin 

# 1min bins 
cor(ds1$totTransVideo, ds1$totTransLogBin, use="complete.obs", method = "pearson") #0.8754
plot(ds1$totTransVideo, ds1$totTransLogBin)
cor(ds1$totTransLogBin, ds1$totTransLog, use="complete.obs")

?cor

#2min bins
cor(ds2$totTransVideo, ds2$totTransLogBin, use="complete.obs") # 0.8745
plot(ds2$totTransVideo, ds2$totTransLogBin)
cor(ds2$totTransLogBin, ds2$totTransLog, use="complete.obs")

#3min bins
cor(ds3$totTransVideo, ds3$totTransLogBin, use="complete.obs") # 0.8996
cor(ds3$totTransLogBin, ds3$totTransLog, use="complete.obs")

#4min bins
cor(ds4$totTransVideo, ds4$totTransLogBin, use="complete.obs") #0.7597
cor(ds4$totTransLogBin, ds4$totTransLog, use="complete.obs")

#5min bins
cor(ds5$totTransVideo, ds5$totTransLogBin, use="complete.obs") #0.6311 
cor(ds5$totTransLogBin, ds5$totTransLog, use="complete.obs")


## Durations per bin
# reorganize the dataset
dsNeu <- subset(ds[,c(1:5, 10:29)])
dsDur <- gather(dsNeu, "ZoneLogBin", "DurationLogBin", c(8, 12, 16, 20, 24))
#dsDur1 <- gather(dsDur, "ZoneVideo", "DurationVideo", c(7, 10, 13, 16, 19))



## try correlation matrix
library("Hmisc")

cm1 <- rcorr(as.matrix(ds1[, 6:29]))
cm1$r #correlations
cm1$P #p values
cm2 <- rcorr(as.matrix(ds2[, 6:29]))



# reorganize correlation matrix
# cormat : matrix of the correlation coefficients
# pmat : matrix of the correlation p-values
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

flattenCorrMatrix(cm1$r, cm1$P)
flattenCorrMatrix(cm2$r, cm2$P)


# visualize correlation matrix
symnum(cm1$r, abbr.colnames = FALSE)

library("corrplot")
corrplot(cm1$r, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)
corrplot(cm2$r, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)


## neue idee: zwei correlation matricen; einmal für durations (alle zonen), einmal für transitions
# mit allen bins in einer correlation -> bin als spalte


dsTrans <- ds[, 1:8]
dsDur <- ds[, c(1:5, 10:29)]


myspread <- function(df, key, value) {
  # quote key
  keyq <- rlang::enquo(key)
  # break value vector into quotes
  valueq <- rlang::enquo(value)
  s <- rlang::quos(!!valueq)
  df %>% gather(variable, value, !!!s) %>%
    unite(temp, !!keyq, variable) %>%
    spread(temp, value)
}

t2 <- dsTrans %>% myspread(Bin, totTransLogBin)


library("Hmisc")

cmTrans <- rcorr(as.matrix(t2[, 5:11]))

library("corrplot")
corrplot(cmTrans$r, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)
CorrTrans <- flattenCorrMatrix(cmTrans$r, cmTrans$P)







##################################################################################

## former tries and ideas
library(lme4)

# Video vs original log
fitLog <- lmer(totTransVideo 
               ~ totTransLog + (1|Pen), data = ds1) 
summary(fitLog)

r2(fitLog)

# video vs binned log
fitLog <- lmer(totTransVideo 
               ~ (Bin + totTransLogBin )^2 + (1|Pen), data = ds) 
summary(fitLog)



fitLog3 <- lmer(totTransVideo 
                ~ totTransLogBin + totTransLog  + (1|Pen), data = ds1) 
summary(fitLog3)
r2(fitLog3)



VarCorr(fitLog)
library("sjstats")
r2(fitLog)

fitLog1 <- lmer(totTransVideo 
                ~ Bin + totTransLogBin + (1|Pen), data = ds) 
summary(fitLog1)

anova(fitLog, fitLog1)
library("multcomp")
summary(glht(fitLog, linfct=mcp(Bin="Tukey")))
summary(glht(fitLog, linfct=mcp(totTransLogBin="Tukey")))


ds$Int<-interaction(ds$totTransLogBin,ds$Bin)
fitLog2<-lmer(totTransVideo~Int + (1|Pen), data=ds )
summary(glht(fitLog2,linfct=mcp(Int="Tukey")))

table(ds$totTransLogBin, ds$Bin)

get_variance(fitLog)
get_variance_fixed(fitLog)
get_variance_residual(fitLog)

hist(resid(fitLog))

r <- resid(fitLog)
hist(r)
windows()
print (summary (fitLog), cor=TRUE)
exp(fixef(fitLog))
par (mfrow= c (3, 3), las= 1)
qqnorm(resid(fitLog, type='response'))
qqnorm(unlist(ranef(fitLog, level=2)))
scatter.smooth(fitted(fitLog), resid(fitLog, type='response'))
boxplot(resid(fitLog, type='response')~ Pen)

?lmer


#
################################################################################################################################

#ATTEMPT AT DISSIMILARITY MEASURES

#creating goldstandard-video dataframe:

goldStandard = data.frame(totTrans = ds$totTransVideo[ds$Bin==1], 
                          dur1 = ds$Dur1Video[ds$Bin==1],dur2 = ds$Dur2Video[ds$Bin==1],
                          dur3 = ds$Dur3Video[ds$Bin==1],dur4 = ds$Dur4Video[ds$Bin==1],
                          dur5 = ds$Dur5Video[ds$Bin==1])
#and bin data frames:
bin1 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==1], 
                  dur1 = ds$Dur1LogBin[ds$Bin==1],dur2 = ds$Dur2LogBin[ds$Bin==1],
                  dur3 = ds$Dur3LogBin[ds$Bin==1],dur4 = ds$Dur4LogBin[ds$Bin==1],
                  dur5 = ds$Dur5LogBin[ds$Bin==1])

bin2 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==2], 
                  dur1 = ds$Dur1LogBin[ds$Bin==2],dur2 = ds$Dur2LogBin[ds$Bin==2],
                  dur3 = ds$Dur3LogBin[ds$Bin==2],dur4 = ds$Dur4LogBin[ds$Bin==2],
                  dur5 = ds$Dur5LogBin[ds$Bin==2])

bin3 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==3], 
                  dur1 = ds$Dur1LogBin[ds$Bin==3],dur2 = ds$Dur2LogBin[ds$Bin==3],
                  dur3 = ds$Dur3LogBin[ds$Bin==3],dur4 = ds$Dur4LogBin[ds$Bin==3],
                  dur5 = ds$Dur5LogBin[ds$Bin==3])

bin4 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==4], 
                  dur1 = ds$Dur1LogBin[ds$Bin==4],dur2 = ds$Dur2LogBin[ds$Bin==4],
                  dur3 = ds$Dur3LogBin[ds$Bin==4],dur4 = ds$Dur4LogBin[ds$Bin==4],
                  dur5 = ds$Dur5LogBin[ds$Bin==4])

bin5 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==5], 
                  dur1 = ds$Dur1LogBin[ds$Bin==5],dur2 = ds$Dur2LogBin[ds$Bin==5],
                  dur3 = ds$Dur3LogBin[ds$Bin==5],dur4 = ds$Dur4LogBin[ds$Bin==5],
                  dur5 = ds$Dur5LogBin[ds$Bin==5])


# A-B diffmatrix

comp1 = as.matrix(goldStandard) - as.matrix(bin1)
comp2 = as.matrix(goldStandard) - as.matrix(bin2)
comp3 = as.matrix(goldStandard) - as.matrix(bin3)
comp4 = as.matrix(goldStandard) - as.matrix(bin4)
comp5 = as.matrix(goldStandard) - as.matrix(bin5)

comp = data.frame(bin = c(1:5), totTrans= integer(5), dur1 = integer(5),dur2 = integer(5),
                  dur3 = integer(5),dur4 = integer(5),dur5 = integer(5))
comp[1,2:7] = apply(comp1, 2, mean)
comp[2,2:7] = apply(comp2, 2, mean)
comp[3,2:7] = apply(comp3, 2, mean)
comp[4,2:7] = apply(comp4, 2, mean)
comp[5,2:7] = apply(comp5, 2, mean)

compMed = data.frame(bin = c(1:5), totTrans= integer(5), dur1 = integer(5),dur2 = integer(5),
                     dur3 = integer(5),dur4 = integer(5),dur5 = integer(5))
compMed[1,2:7] = apply(comp1, 2, median)
compMed[2,2:7] = apply(comp2, 2, median)
compMed[3,2:7] = apply(comp3, 2, median)
compMed[4,2:7] = apply(comp4, 2, median)
compMed[5,2:7] = apply(comp5, 2, median)

# euclidean distance for diffMatrix or directly between
#install.packages('wordspace')
library(wordspace)
d1S = dist.matrix(goldScaled,b1S, method = 'euclidean', byrow=FALSE)
d2S = dist.matrix(goldScaled,b2S, method = 'euclidean', byrow=FALSE)
d3S = dist.matrix(goldScaled,b3S, method = 'euclidean', byrow=FALSE)
d4S = dist.matrix(goldScaled,b4S, method = 'euclidean', byrow=FALSE)
d5S = dist.matrix(goldScaled,b5S, method = 'euclidean', byrow=FALSE)

d1 = dist.matrix(as.matrix(goldStandard),as.matrix(bin1), method = 'euclidean', byrow=FALSE)
d2 = dist.matrix(as.matrix(goldStandard),as.matrix(bin2), method = 'euclidean', byrow=FALSE)
d3 = dist.matrix(as.matrix(goldStandard),as.matrix(bin3), method = 'euclidean', byrow=FALSE)
d4 = dist.matrix(as.matrix(goldStandard),as.matrix(bin4), method = 'euclidean', byrow=FALSE)
d5 = dist.matrix(as.matrix(goldStandard),as.matrix(bin5), method = 'euclidean', byrow=FALSE)

#########
#dataset with all differences

comp.all = data.frame(bin = c(rep(1,54),rep(2,54),rep(3,54),rep(4,54),rep(5,54)),
                      tag = ds$Tag, 
                      pen = ds$Pen)
comp.all = cbind(comp.all, rbind(comp1, comp2, comp3, comp4, comp5))


### standardize along gold standard 
goldScaled = scale(goldStandard)
centerS =    c(5.407407,  172.240741, 1076.092593,  786.888889,  256.962963, 1307.814815) #means gold standard
scaleS =    c(3.988278,  308.698909,  874.493865,  704.802118,  419.729156, 1260.285024) #SD gold standard
b1S = scale(bin1, centerS, scaleS)
b2S = scale(bin2, centerS, scaleS)
b3S = scale(bin3, centerS, scaleS)
b4S = scale(bin4, centerS, scaleS)
b5S = scale(bin5, centerS, scaleS)
# -> only needed if between variables comparisons are necessary or one general value is calculated


# Scaled Diff Matrix

comp1S = as.matrix(goldScaled) - as.matrix(b1S)
comp2S = as.matrix(goldScaled) - as.matrix(b2S)
comp3S = as.matrix(goldScaled) - as.matrix(b3S)
comp4S = as.matrix(goldScaled) - as.matrix(b4S)
comp5S = as.matrix(goldScaled) - as.matrix(b5S)

compS = data.frame(bin = c(1:5), totTrans= integer(5), dur1 = integer(5),dur2 = integer(5),
                   dur3 = integer(5),dur4 = integer(5),dur5 = integer(5))
compS[1,2:7] = apply(comp1S, 2, mean)
compS[2,2:7] = apply(comp2S, 2, mean)
compS[3,2:7] = apply(comp3S, 2, mean)
compS[4,2:7] = apply(comp4S, 2, mean)
compS[5,2:7] = apply(comp5S, 2, mean)

#dataset with all differences

comp.all.S = data.frame(bin = c(rep(1,54),rep(2,54),rep(3,54),rep(4,54),rep(5,54)),
                        tag = ds$Tag, 
                        pen = ds$Pen)
comp.all.S = cbind(comp.all.S, rbind(comp1S, comp2S, comp3S, comp4S, comp5S))

qqnorm(comp.all.S$dur1)
qqline(comp.all.S$dur1)
hist(comp.all.S$dur1)










##### WITH ORIGINAL LOG ###################


rm(list=ls())
library("tidyverse")
ds <- read_delim("Trans_Dur_Comparison_withPen.csv", ";", escape_double = FALSE, trim_ws = TRUE)
ds$Bin <- as.factor(ds$Bin)
ds$Pen <- as.factor(ds$Pen)

names(ds)


# creating goldstandard-video dataframe:

goldStandard = data.frame(totTrans_GB = ds$totTransVideo[ds$Bin==1], 
                          dur1_GB = ds$Dur1Video[ds$Bin==1],
                          dur2_GB = ds$Dur2Video[ds$Bin==1],
                          dur3_GB = ds$Dur3Video[ds$Bin==1],
                          dur4_GB = ds$Dur4Video[ds$Bin==1],
                          dur5_GB = ds$Dur5Video[ds$Bin==1])

# creating original log file dataframe
originalLog = data.frame(totTrans_OB  = ds$totTransLog[ds$Bin==1], 
                         dur1_OB = ds$Dur1Log[ds$Bin==1],
                         dur2_OB = ds$Dur2Log[ds$Bin==1],
                         dur3_OB = ds$Dur3Log[ds$Bin==1],
                         dur4_OB = ds$Dur4Log[ds$Bin==1],
                         dur5_OB = ds$Dur5Log[ds$Bin==1])

#and bin data frames:
bin1 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==1], 
                  dur1 = ds$Dur1LogBin[ds$Bin==1],dur2 = ds$Dur2LogBin[ds$Bin==1],
                  dur3 = ds$Dur3LogBin[ds$Bin==1],dur4 = ds$Dur4LogBin[ds$Bin==1],
                  dur5 = ds$Dur5LogBin[ds$Bin==1])

bin2 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==2], 
                  dur1 = ds$Dur1LogBin[ds$Bin==2],dur2 = ds$Dur2LogBin[ds$Bin==2],
                  dur3 = ds$Dur3LogBin[ds$Bin==2],dur4 = ds$Dur4LogBin[ds$Bin==2],
                  dur5 = ds$Dur5LogBin[ds$Bin==2])

bin3 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==3], 
                  dur1 = ds$Dur1LogBin[ds$Bin==3],dur2 = ds$Dur2LogBin[ds$Bin==3],
                  dur3 = ds$Dur3LogBin[ds$Bin==3],dur4 = ds$Dur4LogBin[ds$Bin==3],
                  dur5 = ds$Dur5LogBin[ds$Bin==3])

bin4 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==4], 
                  dur1 = ds$Dur1LogBin[ds$Bin==4],dur2 = ds$Dur2LogBin[ds$Bin==4],
                  dur3 = ds$Dur3LogBin[ds$Bin==4],dur4 = ds$Dur4LogBin[ds$Bin==4],
                  dur5 = ds$Dur5LogBin[ds$Bin==4])

bin5 = data.frame(totTrans = ds$totTransLogBin[ds$Bin==5], 
                  dur1 = ds$Dur1LogBin[ds$Bin==5],dur2 = ds$Dur2LogBin[ds$Bin==5],
                  dur3 = ds$Dur3LogBin[ds$Bin==5],dur4 = ds$Dur4LogBin[ds$Bin==5],
                  dur5 = ds$Dur5LogBin[ds$Bin==5])


# A-B diffmatrix

comp_GB_1 = as.matrix(goldStandard) - as.matrix(bin1)
comp_GB_2 = as.matrix(goldStandard) - as.matrix(bin2)
comp_GB_3 = as.matrix(goldStandard) - as.matrix(bin3)
comp_GB_4 = as.matrix(goldStandard) - as.matrix(bin4)
comp_GB_5 = as.matrix(goldStandard) - as.matrix(bin5)

comp_GO = as.matrix(goldStandard) - as.matrix(originalLog)


#dataset with all differences

comp.all.prep.bin = data.frame(bin = c(rep(1,54),rep(2,54),rep(3,54),rep(4,54),rep(5,54)),
                           tag = ds$Tag, 
                           pen = ds$Pen,
                           date = ds$Date)

comp.all.prep.org = data.frame(bin = c(rep("orig",54)),
                               tag = ds$Tag[ds$Bin==1], 
                           pen = ds$Pen[ds$Bin==1],
                           date = ds$Date[ds$Bin==1])

# create actual file:
diff.all.GB = rbind(comp_GB_1, comp_GB_2, comp_GB_3, comp_GB_4, comp_GB_5)
diff.all..GB = cbind(comp.all.prep.bin, diff.all.GB)

diff.GO = cbind(comp.all.prep.org, comp_GO)

diff.all=rbind(diff.all..GB, diff.GO)







#### plots ####
ggplot(diff.all, aes(totTrans_GB, colour = factor(bin))) + 
  geom_density()+
  xlab('Difference to video') + ylab('Density')+
  ggtitle('Total number of transitions')+
  geom_vline(xintercept = 0, linetype = 'dashed') +
  theme_bw() +
  theme(axis.line = element_line(colour = "black")) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank()) +
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=18,face="bold")) +
  theme(legend.text = element_text(size = 12),
        legend.title = element_text(size = 18, face="bold"))

ggsave(filename = paste0("trans_density_nice.png"), plot = last_plot(), 
       width = 20, height = 15, dpi = 300, units = "cm")



ggplot(diff.all, aes(totTrans_GB, fill = bin, color = bin)) + 
  geom_histogram(position = "dodge", alpha = 0.2, bins = 50)+
  xlab('Difference to video') + ylab('Frequency')+
  ggtitle('Total number of transitions')+
  geom_vline(xintercept = 0, linetype = 'dashed') +
  theme_bw() +
  theme(axis.line = element_line(colour = "black")) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank()) +
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=18,face="bold")) +
  theme(legend.text = element_text(size = 12),
        legend.title = element_text(size = 18, face="bold"))

ggsave(filename = paste0("trans_hist_nice.png"), plot = last_plot(), 
       width = 20, height = 15, dpi = 300, units = "cm")




#### models ####

### Transitions ###



### model with poisson
library(lme4)

hist(abs(diff.all$totTrans_GB))
fit.abs <- glmer(abs(totTrans_GB) ~ bin +(1|pen/tag),family=poisson,data=diff.all) 
fit.absNull <- glmer(abs(totTrans_GB) ~ (1|pen/tag),family=poisson,data=diff.all) 
anova(fit.abs, fit.absNull)
library(multcomp)
summary(glht(fit.abs, linfct= mcp(bin= 'Tukey')), test = adjusted('bonferroni'))

library(DHARMa)
resid.df<- simulateResiduals(fit.abs, 1000)
plotSimulatedResiduals(resid.df)

# exclude bin size 3
diff.all_2 <- subset(diff.all, diff.all$bin != 3)
unique(diff.all_2$bin)

library(lme4)
hist(abs(diff.all_2$totTrans_GB))
fit.abs <- glmer(abs(totTrans_GB) ~ bin +(1|pen/tag),family=poisson,data=diff.all_2) 
fit.absNull <- glmer(abs(totTrans_GB) ~ (1|pen/tag),family=poisson,data=diff.all_2) 
anova(fit.abs, fit.absNull)
library(multcomp)
summary(glht(fit.abs, linfct= mcp(bin= 'Tukey')), test = adjusted('bonferroni'))

library(DHARMa)
resid.df<- simulateResiduals(fit.abs, 1000)
plotSimulatedResiduals(resid.df)


# plot 
ggplot(diff.all_2, aes(totTrans_GB, colour = factor(bin))) + 
  geom_density()+
  xlab('Difference to gold standard') + ylab('Density')+
  ggtitle('Total number of transitions')+
  geom_vline(xintercept = 0, linetype = 'dashed')





## durations
# reorganize the data
Neu <- subset(diff.all[,c(1:4, 6:10)])
diff.all.Dur <- gather(Neu, "Zone", "DurationDiff_GB", c(5:9))
#diff.all.Dur.GB$Zone <- as.factor(gsub('\\_.*', '', diff.all.Dur.GB$Zone))
diff.all.Dur$Zone <- as.factor(substr(diff.all.Dur$Zone, 4, 5))
diff.all.Dur$bin <- as.factor(diff.all.Dur$bin)


# Data distribution
qqnorm(diff.all.Dur$DurationDiff_GB)
qqline(diff.all.Dur$DurationDiff_GB)
hist(diff.all.Dur$DurationDiff_GB)
hist(abs(diff.all.Dur$DurationDiff_GB))

# model with poisson
library(lme4)
fit.absDur <- glmer(abs(DurationDiff_GB) ~ bin 
                    +(1|pen/tag),
                    family=poisson,
                    data=diff.all.Dur) 

library(DHARMa)
resid.df<- simulateResiduals(fit.absDur, 1000)
plotSimulatedResiduals(resid.df)
# nö



### binomial 
# match vs no match in durations (zero vs not zero in difference)
diff.all.Dur$Binom <- diff.all.Dur$DurationDiff_GB
diff.all.Dur$Binom[diff.all.Dur$Binom != 0] <- 1


# for the bin comparison, the zone is actually not relevant,
# as it should work for all zones
modDur <- glmer(Binom ~ bin 
                + (1|pen/tag), data = diff.all.Dur,
                family = binomial)
# Achtung: singular fit

# reduce bin (creating a NULL Model)
modDur1 <- update(modDur, .~.-bin)
anova(modDur, modDur1)



# everything without bin 3

# reorganize the data
Neu <- subset(diff.all_2[,c(1:4, 6:10)])
diff.all.Dur_2 <- gather(Neu, "Zone", "DurationDiff_GB", c(5:9))
#diff.all.Dur.GB$Zone <- as.factor(gsub('\\_.*', '', diff.all.Dur.GB$Zone))
diff.all.Dur_2$Zone <- as.factor(substr(diff.all.Dur_2$Zone, 4, 5))
diff.all.Dur_2$bin <- as.factor(diff.all.Dur_2$bin)
unique(diff.all.Dur_2$bin)

fit.absDur <- glmer(abs(DurationDiff_GB) ~ bin 
                    +(1|pen/tag),
                    family=poisson,
                    data=diff.all.Dur_2) 

library(DHARMa)
resid.df<- simulateResiduals(fit.absDur, 1000)
plotSimulatedResiduals(resid.df)




#### binomial 
# match vs no match in durations (zero vs not zero in difference)
diff.all.Dur_2$Binom <- diff.all.Dur_2$DurationDiff_GB
diff.all.Dur_2$Binom[diff.all.Dur_2$Binom != 0] <- 1


# for the bin comparison, the zone is actually not relevant,
# as it should work for all zones
modDur <- glmer(Binom ~ bin 
                + (1|pen/tag), data = diff.all.Dur_2,
                family = binomial)
# Achtung: singular fit

# reduce bin (creating a NULL Model)
modDur1 <- update(modDur, .~.-bin)
anova(modDur, modDur1)


library(multcomp)
summary(glht(modDur, linfct= mcp(bin= 'Tukey')), test = adjusted('bonferroni'))

unique(diff.all.Dur_2$bin)
table(diff.all.Dur_2$bin, diff.all.Dur_2$Binom)



# plot
ggplot(diff.all.Dur_2, aes(DurationDiff_GB, colour = factor(bin))) + 
  geom_density()+
  xlab('Difference to gold standard') + ylab('Density')+
  ggtitle('Duration irrespective of zone')+
  geom_vline(xintercept = 0, linetype = 'dashed') +
  xlim(-200, 200)

hist()





#### ABS DIFFERENCE ####
diff.all_2 <- subset(diff.all, diff.all$bin != 3)

## transitions
table(diff.all_2$bin)

TransGB_desc <- diff.all_2 %>%
  group_by(bin) %>%
  summarise(mean_Trans = mean(totTrans_GB), 
            SD_Trans = sd(totTrans_GB),
            #n_Trans = n()) #for some awkward reason not working
            n_Trans = c(54))%>%
  mutate(se_Trans = SD_Trans / sqrt(n_Trans),
         lower.CI = mean_Trans - qt(1 - (0.05 / 2), n_Trans - 1) * se_Trans,
         upper.CI = mean_Trans + qt(1 - (0.05 / 2), n_Trans - 1) * se_Trans)

TransGB_desc$bin <- as.factor(TransGB_desc$bin)
levels(TransGB_desc$bin)
library(plyr)
TransGB_desc$bin <- revalue(TransGB_desc$bin, c("orig" = "0"))
TransGB_desc$bin <- relevel(TransGB_desc$bin, "0")

# plot
pd <-  position_dodge(0.2)

ggplot(TransGB_desc, aes(x=bin, y=mean_Trans)) + 
  geom_errorbar(aes(ymin=lower.CI, ymax=upper.CI), position=pd, width=.5) + 
  geom_point(position=pd)+
  geom_abline(intercept=0, slope=0, linetype="dashed") +
  ylab("mean Transitions +/- 95% CI") +
  theme_bw() +
  theme(axis.line = element_line(colour = "black")) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank()) +
  theme(axis.text=element_text(size=14),
        axis.title=element_text(size=16,face="bold"))

ggsave(filename = "transitions_mean_CI.png", plot = last_plot(), 
       width = 20, height = 15, dpi = 300, units = "cm")





# reorganize the data
# GB
Neu <- subset(diff.all_2[,c(1:4, 6:10)])
diff.all.Dur <- gather(Neu, "Zone", "DurationDiff_GB", c(5:9))
#diff.all.Dur.GB$Zone <- as.factor(gsub('\\_.*', '', diff.all.Dur.GB$Zone))
diff.all.Dur$Zone <- as.factor(substr(diff.all.Dur$Zone, 4, 5))

diff.all.Dur$bin <- as.factor(diff.all.Dur$bin)


# for Durations Goldstandard-binned
table(diff.all.Dur$bin)

DurGB_desc <- diff.all.Dur %>%
  group_by(bin) %>%
  summarise(mean_Dur = mean(DurationDiff_GB), 
            SD_Dur = sd(DurationDiff_GB),
            #n_Trans = n()) #for some awkward reason not working
            n_Dur = c(270))%>%
  mutate(se_Dur = SD_Dur / sqrt(n_Dur),
         lower.CI = mean_Dur - qt(1 - (0.05 / 2), n_Dur - 1) * se_Dur,
         upper.CI = mean_Dur + qt(1 - (0.05 / 2), n_Dur - 1) * se_Dur)


# plot
pd <-  position_dodge(0.2)
ggplot(DurGB_desc, aes(x=bin, y=mean_Dur)) + 
  geom_errorbar(aes(ymin=lower.CI, ymax=upper.CI), position=pd, width=.5) + 
  geom_point(position=pd)+
  geom_abline(intercept=0, slope=0, linetype="dashed") +
  ylab("mean Durations +/- 95% CI")




### reliability ######
rm(list=ls())
library("tidyverse")
ds1 <- read_delim("Trans_Dur_Comparison_withPen.csv", ";", escape_double = FALSE, trim_ws = TRUE)
ds1$Bin <- as.factor(ds1$Bin)
ds1$Pen <- as.factor(ds1$Pen)

library(epiR)
library(DescTools)

names(ds1)


# original vs video
ds_1 <- subset(ds1, ds1$Bin==1)
names(ds_1)
unique(ds_1$Bin)

# original
hist(ds_1$totTransLog)
hist(ds_1$totTransVideo)
CCC(ds_1$totTransLog, ds_1$totTransVideo, ci = "z-transform", conf.level = 0.95, na.rm = FALSE)

library(irr)
x <- as.data.frame(cbind(ds_1$totTransLog, ds_1$totTransVideo))
icc(x, model = "twoway", 
    type = "agreement")

# 1min bin
hist(ds_1$totTransLogBin)
CCC(ds_1$totTransLogBin, ds_1$totTransVideo, ci = "z-transform", conf.level = 0.95, na.rm = FALSE)

# 5min bin
ds_5 <- subset(ds1, ds1$Bin == 5)
hist(ds_5$totTransLogBin)
CCC(ds_5$totTransLogBin, ds_5$totTransVideo, ci = "z-transform", conf.level = 0.95, na.rm = FALSE)




# durations
names(ds_1)
dsNeuLog <- subset(ds_1[,c(1:5, 10, 14, 18, 22, 26)])
dsNeuVideo <- subset(ds_1[,c(1:5, 11, 15, 19, 23, 27)])

dsDurLog <- gather(dsNeuLog, "Zone", "Duration_Log", c(6:10))
dsDurLog$Zone <- as.factor(substr(dsDurLog$Zone, 4, 4))
dsDurVideo <- gather(dsNeuVideo, "Zone", "Duration_Video", c(6:10))
dsDurVideo$Zone <- as.factor(substr(dsDurVideo$Zone, 4, 4))
dsDur <- merge(dsDurLog, dsDurVideo, by = c("Date", "BirdID", "Pen", "Zone"), )


hist(dsDur$Duration_Log)
hist(dsDur$Duration_Video)
CCC(dsDur$Duration_Log, dsDur$Duration_Video, ci = "z-transform", conf.level = 0.95, na.rm = FALSE)



# 1min bin
names(ds_1)
dsNeuLogBin <- subset(ds_1[,c(1:5, 12, 16, 20, 24, 28)])
dsNeuVideo <- subset(ds_1[,c(1:5, 11, 15, 19, 23, 27)])

dsDurLog <- gather(dsNeuLogBin, "Zone", "Duration_LogBin", c(6:10))
dsDurLog$Zone <- as.factor(substr(dsDurLog$Zone, 4, 4))
dsDurVideo <- gather(dsNeuVideo, "Zone", "Duration_Video", c(6:10))
dsDurVideo$Zone <- as.factor(substr(dsDurVideo$Zone, 4, 4))
dsDur <- merge(dsDurLog, dsDurVideo, by = c("Date", "BirdID", "Pen", "Zone"), )


hist(dsDur$Duration_LogBin)
hist(dsDur$Duration_Video)
CCC(dsDur$Duration_LogBin, dsDur$Duration_Video, ci = "z-transform", conf.level = 0.95, na.rm = FALSE)






# include original logs as bin = 0
ds2 <- subset(ds1, ds1$Bin=="1")
ds2$Bin <- "0" # name bin 0 for original files
ds <- rbind(ds1, ds2)


# create subsets per bin
#bins = c(0, 5, 10, 15, 20, 30, 40, 50, 60, 120, 240, 300)
bins = c(0, 1, 2, 4, 5)

#bin=bins[1]

for (bin in bins) {
  assign(paste0("bin_", bin),
         data.frame(date = ds$Date[ds$Bin==bin],
                    BirdID = ds$BirdID[ds$Bin==bin],
                    pen = ds$Pen[ds$Bin==bin],
                    totTrans_Log = ds$totTransLogBin[ds$Bin==bin],
                    totTrans_Vid = ds$totTransVideo[ds$Bin==bin],
                    dur1_Log = ds$Dur1LogBin[ds$Bin==bin],
                    dur1_Vid = ds$Dur1Video[ds$Bin==bin],
                    dur2_Log = ds$Dur2LogBin[ds$Bin==bin],
                    dur2_Vid = ds$Dur2Video[ds$Bin==bin],
                    dur3_Log = ds$Dur3LogBin[ds$Bin==bin],
                    dur3_Vid = ds$Dur3Video[ds$Bin==bin],
                    dur4_Log = ds$Dur4LogBin[ds$Bin==bin],
                    dur4_Vid = ds$Dur4Video[ds$Bin==bin],
                    dur5_Log = ds$Dur5LogBin[ds$Bin==bin],
                    dur5_Vid = ds$Dur5Video[ds$Bin==bin]))
}



# create list of bin dataframes
BinFiles<-grep("bin_",names(.GlobalEnv),value=TRUE)
BinFiles_list<-do.call("list",mget(BinFiles))



# Lin's concordance correlation coefficient 
# works with small sample sizes, and makes fewer assumptions about distribution than the ICC.
# should be stable for poisson distribution too

library(epiR)
library(DescTools)

#i=1



for (i in 1:length(BinFiles_list)) {
  # CCC for transitions
  x <- CCC(BinFiles_list[[i]]$totTrans_Log, BinFiles_list[[i]]$totTrans_Vid, ci = "z-transform", conf.level = 0.95, na.rm = FALSE)
  print(names(BinFiles_list)[i]) # print name of the dataframe (when using BinFiles_list[[1]] it lists values)
  # extract values
  LCCC_T = round(as.numeric(x$rho.c[1]), digits=3)
  l.CI_T = round(as.numeric(x$rho.c[2]), digits=3)
  u.CI_T = round(as.numeric(x$rho.c[3]), digits=3)
  
  # reorganize dataset for durations
  dsNeuLog <- subset(BinFiles_list[[i]][,c(1:3, 6, 8, 10, 12, 14)])
  dsNeuVideo <- subset(BinFiles_list[[i]][,c(1:3, 7, 9, 11, 13, 15)])
  
  dsDurLog <- gather(dsNeuLog, "Zone", "Duration_Log", c(4:8))
  dsDurLog$Zone <- as.factor(substr(dsDurLog$Zone, 3, 4))
  dsDurVideo <- gather(dsNeuVideo, "Zone", "Duration_Video", c(4:8))
  dsDurVideo$Zone <- as.factor(substr(dsDurVideo$Zone, 3, 4))
  dsDur <- merge(dsDurLog, dsDurVideo, by = c("date", "BirdID", "pen", "Zone"), )
  #hist(dsDur$Duration_Log)
  #hist(dsDur$Duration_Video)
  
  # CCC for durations
  y <- CCC(dsDur$Duration_Log, dsDur$Duration_Video, ci = "z-transform", conf.level = 0.95, na.rm = FALSE)
  # extract values
  LCCC_D = round(as.numeric(y$rho.c[1]), digits=3)
  l.CI_D = round(as.numeric(y$rho.c[2]), digits=3)
  u.CI_D = round(as.numeric(y$rho.c[3]), digits=3)
  
  # output
  print(paste("Transitions CCC:", LCCC_T))
  print(paste("lwr CI:", l.CI_T))
  print(paste("upr CI:", u.CI_T))
  
  print(paste("Durations CCC:", LCCC_D))
  print(paste("lwr CI:", l.CI_D))
  print(paste("upr CI:", u.CI_D))
}




##### END ##########
