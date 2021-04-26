
library(lubridate)

#clear workspace
rm(list = ls())

#set working directory
#setwd("G:/VPHI/Welfare/2- Research Projects/Laura C-Klara G/tracking system/binning_validation/AVIFORUM 29-30")
setwd("//nas-vetsuisse/vetsuisse/Gruppen/VPHI/Welfare/2- Research Projects/Laura C-Klara G/tracking system/binning_validation")

#extract all file names in folder
all_files = list.files(path = paste0(getwd(),"/direct comparison"))
id1 = read.csv("videoBinningProcess/tagID_backpack_Legring_pen_2019_11_11.csv", header = TRUE, sep = ';', fileEncoding="UTF-8-BOM")
id2 = read.csv("videoBinningProcess/tagID_backpack_Legring_pen_2020_01_21.csv", header = TRUE, sep = ';', fileEncoding="UTF-8-BOM")
#View(id2)

bins = c(1, 2, 3, 4, 5, 10, 15, 20)
days = c('13', '1', '9')
hens = c('3gp', '3pb', '3pp', '4sp', '4ss', '4sws', '5ps', '5wp', '5ws', 
         '10pp', '10ss', '10ps', '11gp', '11pb', '11sp', '12ps', '12wp', '12ws')

comparison_Bins = data.frame(bin = character(0), 
                             match = integer(0),
                             no_match = integer(0),
                             tot_comp = integer(0),
                             pen = character(0),
                             day = character(0),
                             hen = character(0))

for (bin in bins){
  
  binData = read.csv(paste0(getwd(),"/direct comparison/", 
                            all_files[grep(paste0('bin',bin,'.csv'), all_files)]), 
                            header = TRUE,sep = ';')
  
  for (pen in c(3,4,5,10,11,12)){
    
    binDataRel = binData[grep(paste(pen), binData$hen),]
    binDataRel$timestamp = as_date(binDataRel$timestamp)
    henRel = hens[grep(paste(pen), hens)]
    
    
    for (day in days){
    
    binDataPen = binDataRel[day(binDataRel$timestamp) == day,]         
    
    for (hen in henRel){
    
      binDataHen = binDataPen[binDataPen$hen == hen,]         
      
      #number of observations
      all = length(binDataHen$hen)
      #number correct classfied instances
      true = sum(binDataHen$tracking == binDataHen$video)
      #number false classified instances
      false = sum(binDataHen$tracking != binDataHen$video)
      
      timePoint = binDataHen$timestamp[1] 
      
      new_entry = data.frame(bin = bin, 
                             match = true,
                             no_match = false,
                             tot_comp = all,
                             pen = pen,
                             day = timePoint,
                             hen = hen)
      comparison_Bins = rbind(comparison_Bins, new_entry)
      }
    }
  }
  
}

comparison_Bins$bin = factor(comparison_Bins$bin)
comparison_Bins$pen = factor(comparison_Bins$pen)
comparison_Bins$day = factor(comparison_Bins$day)
View(comparison_Bins)

################################################
#auswertung

library(lme4)
fit <- glmer(cbind(comparison_Bins$match, 
                   comparison_Bins$no_match) 
             ~ bin + day + (1|pen), data = comparison_Bins, family = binomial) 
#WHY NO-MATCH INSTEAD OF TOTAL??

fit_more <- glmer(cbind(comparison_Bins$match, 
                    comparison_Bins$no_match) 
              ~ bin + day + bin*day + (1|pen/hen), data = comparison_Bins, family = binomial) 
# the way the model is established it consideres the "offset" having different numbers per bin to calculate the percentage of matches
#Residuen testen
library(DHARMa)
bin.resid.df<- simulateResiduals(fit_more, 1000)
plotSimulatedResiduals(bin.resid.df)
plot(bin.resid.df)



# Modellvergleich mit Nullmodel
fit_more_NULL <- glmer(cbind(comparison_Bins$match, 
                             comparison_Bins$no_match) 
                       ~ 1 + (1|pen/hen), data = comparison_Bins, family = binomial) 
anova(fit_more, fit_more_NULL) # full model is better


# Modellvergleich mit vs ohne interaction
fit_noInt <- glmer(cbind(comparison_Bins$match, 
                   comparison_Bins$no_match) 
             ~ bin + day + (1|pen/hen), data = comparison_Bins, family = binomial) 
anova(fit_more, fit_noInt) #without interaction is better

# Modellvergleich mit vs ohne bin
fit3 <- glmer(cbind(comparison_Bins$match, 
                   comparison_Bins$no_match) 
             ~ day + (1|pen/hen), data = comparison_Bins, family = binomial) 
anova(fit_noInt, fit3) #with bin

# Modellvergleich mit vs ohne day
fit4 <- glmer(cbind(comparison_Bins$match, 
                   comparison_Bins$no_match) 
             ~ bin + (1|pen/hen), data = comparison_Bins, family = binomial) 
anova(fit_noInt, fit4) # with day

## => best model is with bin and day but without interaction

anova(fit_more, fit4)
anova(fit_noInt, fit_more_NULL)

library(multcomp)

summary(glht(fit_noInt, linfct=mcp(bin="Tukey", day = 'Tukey')))
# pairwise comparison shows: no significant different between bin 1,2,3,4 and 5 ==> so all 5 are choosable
# but because we assume the smaller the bin the less data loss, we would go for bin 1
best.match <- aggregate(comparison_Bins$match/comparison_Bins$tot_comp,by=list(comparison_Bins$bin), FUN=mean)
print(best.match)
# ==> bin 1 (94.4%), 2 (94.2 %) and 3 (94.8%) have almost identically high matches and are not 
# significantly different from each other according to the glmer model
# therefore, we can go for 1 min bins :-)

#Residuen testen
library(DHARMa)
bin.resid.df<- simulateResiduals(fit_noInt, 1000)
plotSimulatedResiduals(bin.resid.df)
plot(bin.resid.df)
# looks go and fullfilles the normality


#### plots
##########################################################################
library(ggplot2)
# All comparisons
tmp <- as.data.frame(confint(glht(fit, mcp(bin = "Tukey")))$confint)
tmp$Comparison <- rownames(tmp)
ggplot(tmp, aes(x = Comparison, y = Estimate, ymin = lwr, ymax = upr)) +
  geom_errorbar() + geom_point()


# Pen 12 is worst case of all Pens: (do we know why? e.g. most flickerings?)
ggplot(comparison_Bins, aes(x = bin, y = match/tot_comp,col=pen)) + geom_point()
ggplot(comparison_Bins, aes(x = bin, y = match/tot_comp)) + geom_boxplot()

# single factor effects
tmp <- as.data.frame(confint(glht(fit))$confint)
tmp$Comparison <- rownames(tmp)
ggplot(tmp, aes(x = Comparison, y = Estimate, ymin = lwr, ymax = upr)) +
  geom_errorbar() + geom_point()

# single factor effects
tmp <- as.data.frame(confint(glht(fit))$confint)
tmp$Comparison <- rownames(tmp)
ggplot(tmp, aes(x = Comparison, y = Estimate, ymin = lwr, ymax = upr)) +
  geom_errorbar() + geom_point()














### old version #################################################################################################
#auswertung

library(lme4)
fit <- glmer(cbind(comparison_Bins$match, 
                   comparison_Bins$no_match) 
             ~ bin + day + (1|pen), data = comparison_Bins, family = binomial) 
#WHY NO-MATCH INSTEAD OF TOTAL??

fit_more <- glmer(cbind(comparison_Bins$match, 
                        comparison_Bins$no_match) 
                  ~ bin + day + (1|pen/hen), data = comparison_Bins, family = binomial) 
# the way the model is established it consideres the "offset" having different numbers per bin to calculate the percentage of matches
summary(fit)

fit2 <- glmer(cbind(comparison_Bins$match, 
                    comparison_Bins$no_match) 
              ~ bin + (1|pen), data = comparison_Bins, family = binomial) 
# the way the model is established it consideres the "offset" having different numbers per bin to calculate the percentage of matches
summary(fit2)

fit3 <- glmer(cbind(comparison_Bins$match, 
                    comparison_Bins$no_match) 
              ~ day + (1|pen), data = comparison_Bins, family = binomial) 
# the way the model is established it consideres the "offset" having different numbers per bin to calculate the percentage of matches
summary(fit3)

fit4 <- glmer(cbind(comparison_Bins$match, 
                    comparison_Bins$no_match) 
              ~ 1 + (1|pen), data = comparison_Bins, family = binomial) 
# the way the model is established it consideres the "offset" having different numbers per bin to calculate the percentage of matches
summary(fit4)

fit5 <- glmer(cbind(comparison_Bins$match, 
                    comparison_Bins$no_match) 
              ~ bin*day + (1|pen), data = comparison_Bins, family = binomial) 
# the way the model is established it consideres the "offset" having different numbers per bin to calculate the percentage of matches
summary(fit5)

fit_more_ohne <- glmer(cbind(comparison_Bins$match, 
                             comparison_Bins$no_match) 
                       ~ 1 + (1|pen/hen), data = comparison_Bins, family = binomial) 



# Modellvergleich mit Nullmodel
anova(fit_more, fit_more_ohne) # full model is better

an
#fit is the best (interaction not relevant)

library(multcomp)

summary(glht(fit_more, linfct=mcp(bin="Tukey", day = 'Tukey')))
# pairwise comparison shows: no significant different between bin 1,2,3,4 and 5 ==> so all 5 are choosable
# but because we assume the smaller the bin the less data loss, we would go for bin 1
best.match <- aggregate(comparison_Bins$match/comparison_Bins$tot_comp,by=list(comparison_Bins$bin), FUN=mean)
print(best.match)
# ==> bin 1 (94.4%), 2 (94.2 %) and 3 (94.8%) have almost identically high matches and are not 
# significantly different from each other according to the glmer model
# therefore, we can go for 1 min bins :-)
#Residuen testen
library(DHARMa)
bin.resid.df<- simulateResiduals(fit, 1000)
plotSimulatedResiduals(bin.resid.df)
plot(bin.resid.df)
# looks go and fullfilles the normality

