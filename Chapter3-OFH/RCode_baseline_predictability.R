# predictability calculation during testperiod ( 1 day before until last test day)

# Laptop.YG
#NO.df <- read.table ('//nas-vetsuisse/VETSUISSE/Benutzer/yg18q990/Project_BLVMethods/NO_NE_Analysis/NO_trackEval_new.csv', header= TRUE, sep= ',')
#Ubelix
#NO.df <- read.table ('/storage/homefs/yg18q990/NO_NE_Analysis/NO_trackEval_new.csv', header= TRUE, sep= ',',stringsAsFactors = FALSE)
# Ubelix_Chicken_Run
NO.df <- read.table ('/storage/research/vetsuisse_chicken_run/NO_NE_Analysis/NO_trackEval_new.csv', header= TRUE, sep= ',',stringsAsFactors = FALSE)

NO.df$X <- NULL

# preparations #
library("data.table")
library("tidyverse")
library("lme4")
#library("lmerTest")
library("broom")
library("brms")
library("multcomp")
library("broom.mixed")
#library("plotly")

str(NO.df)
NO.df$Event <- as.factor(NO.df$Event)
NO.df$Pen <- as.factor(NO.df$Pen)
NO.df$BirdID <- as.factor(NO.df$BirdID)
NO.df$Zone <- as.factor(NO.df$Zone)
NO.df$firstVisit <- as.numeric(NO.df$firstVisit)
NO.df$firstVisitDur <- as.numeric(NO.df$firstVisitDur)

##############################################################################

NO.baseline.df <- subset(NO.df, Event == "baseline" | Event =="vaccination")
#View(NO.test.df)

# add days as a Dateproxy

Dates <- as.data.frame(sort(unique(NO.baseline.df$Date)))
Dates$day <- rownames(Dates)
names(Dates)[1] <- "Date"
str(Dates)
baseline_day <- merge(Dates,NO.baseline.df, by = "Date",all=T)
str(baseline_day)
baseline_day$day <- as.numeric(baseline_day$day)


# polynomial needed?
library(lme4)
#library(lmerTest)
## firstVisit = latency ####
fit <- lmer(firstVisit~poly(day,2)+Event+(poly(day,2)|BirdID),data=baseline_day)
anova(fit)

# Residuenanalysen #
par (mfrow= c (3, 2))
qqnorm (resid (fit))
qqnorm (ranef (fit) [['BirdID']] [, 1])
scatter.smooth (fitted (fit), resid (fit))
boxplot (split (resid (fit), baseline_day [, 'Pen']))

plot(fit)

fit.2 <- lmer(firstVisit~poly(day,1)+Event +(poly(day,1)|BirdID),data=baseline_day)
summary(fit.2)
anova(fit.2)

anova(fit,fit.2)

# we don't need to keep the poly of day in the model if considering all zones

# in case of Litter focus only:
fit.3 <- lmer(firstVisit~poly(day,2)+Event+(poly(day,2)|BirdID),data=subset(baseline_day,Zone=="Litter"))
anova(fit.3)

fit.4 <- lmer(firstVisit~poly(day,1)+Event+(poly(day,1)|BirdID),data=subset(baseline_day,Zone=="Litter"))
anova(fit.4,fit.3)

# we need to keep poly within the model

baseline_day$day <- as.integer(baseline_day$day)

# behavioral predictability all zones together
#### double hierarchical model ####

# behavioral predictability all zones together
#### double hierarchical model ####
# Variation in behavioral predictability
library(brms)
double_model = bf(firstVisit ~poly(day,2) + Pen + (poly(day,2)|BirdID),
                  sigma ~(1|BirdID))
library(parallel)
my.cores <- detectCores()
m3_brm <- brm(double_model, data=subset(baseline_day,Zone=="Litter"),
              warmup = 5000, iter=100000,thin=2,
              chains=4,inits="random",
              seed=12345,
              core=my.cores)

save.image(file="m3_brm_baseline_firstVisit_litter.Rdata")


# Variation in behavioral predictability
library(brms)
double_model = bf(firstVisit ~poly(day,1) + Event + (poly(day,1)|BirdID),
                  sigma ~(1|BirdID))
library(parallel)
my.cores <- detectCores()
m3_brm <- brm(double_model, data=baseline_day,
              warmup = 5000, iter=100000,thin=2,
              chains=4,inits="random",control = list(adapt_delta = 0.999,max_treedepth = 20),
              seed=12345,
              core=my.cores)

save.image(file="m3_brm_baseline_firstVisit_allzones.Rdata")
