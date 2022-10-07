library(brms) #predictability: Double hierearchical model
library(MCMCglmm) #HPDinterval
library(dplyr) #%>%
library(lmerTest) #lmer
#library(MuMIn) #r2
#library(merTools) #sim
library(sjPlot) #plot coeff
library(ggplot2) #ylim in plot
library(rstan) #model diagnostic

library(DHARMa) #sim
library(glmmTMB) 


####################### Download data
#path_ = 'G:/VPHI/Welfare/2- Research Projects/OFHE2.OriginsE2/DataOutput/TrackingSystem/ALLDATA_'
path_ = '/storage/research/vetsuisse_chicken_run/OriginsProject/Dirichlet'
#df = read.csv(file.path(path_,'df_daily_aggregatedHA.csv'), header = TRUE, sep = ",")
df = read.csv(file.path(path_,'OFH_df_all_mvt.csv'), header = TRUE, sep = ",")
df$HenID = as.factor(df$HenID)   
df$PenID = as.factor(df$PenID)
df$level = as.factor(df$level) 
df$CLASS = as.factor(df$CLASS) 
df$Treatment = as.factor(df$Treatment)
df$time = scale(df$WIB, center=FALSE, scale=sd(df$WIB, na.rm = TRUE)) #as not centering, we need to add this
df$time2_ = poly(df$time, degree=2,raw=TRUE)[,2]
df$time2 = scale(df$time2_, center=FALSE, scale=sd(df$time2_, na.rm = TRUE))
#df$KBF_interp = scale(df$KBF_interp, center=TRUE)
#df$FeatherDamage_interp = scale(df$FeatherDamage_interp, center=TRUE)
#df$weight_interp = scale(df$weight_interp, center=TRUE)

#scale: mean-centering of the environmental variable so that intercepts reflect average values for the HenID and ind. 
#df$temperature_C_avg_scale = scale(df$temperature_C_avg, center=TRUE, scale=TRUE)

#setting reference group
df <- df %>% mutate(Treatment = relevel(Treatment, ref = "TRAN"))
contrasts(df$Treatment)
df <- df %>% mutate(CLASS = relevel(CLASS, ref = "REXP"))
contrasts(df$CLASS)