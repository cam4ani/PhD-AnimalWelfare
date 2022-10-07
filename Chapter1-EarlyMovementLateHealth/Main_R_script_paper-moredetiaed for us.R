library(psych) #PCA
library(brms) #predictability: Double hierearchical model
library(MCMCglmm) #HPDinterval
library(dplyr) #%>%
library(parallel) #several cores in brms
library(lme4)
library(lmerTest) #lmer with pvalues
library(MuMIn) #R2
library(merTools) #sim
library(dplyr) #%>%
library(MCMCglmm) #bivariate model
library(broom)
library(nadiv)
library(tidyverse)

my.cores = detectCores() #speed up

#download data
df = read.csv('df_MVT_4individuality.csv', header = TRUE, sep = ";")
df$HenID = as.factor(df$HenID)   
df$PenID = as.factor(df$PenID) 
df$cDIB = as.integer(df$DIB_startat11)
df$CLASS = as.factor(df$CLASS) 
df$Treatment = as.factor(df$Treatment)
df$temperature_C_avg_scale = scale(df$temperature_C_avg, center=TRUE, scale=TRUE)
df$time = scale(df$cDIB, center=FALSE, scale=sd(df$cDIB, na.rm = TRUE))
df$time2 = poly(df$time, degree=2, raw=TRUE)[,2]
df$avgDIB_scale = scale(df$avgDIB, center=TRUE, scale=TRUE)
df$InitialWeight_scale = scale(df$InitialWeight, center=TRUE, scale=TRUE)
#setting reference group
contrasts(df$Treatment)
df <- df %>% mutate(Treatment = relevel(Treatment, ref = "TRAN"))
contrasts(df$Treatment)
contrasts(df$CLASS)
df <- df %>% mutate(CLASS = relevel(CLASS, ref = "LEXP"))
contrasts(df$CLASS)
print(dim(df))
summary(df)

################################################## PCA ##################################################
li_pca = c('perc_duration_5_Zone','perc_duration_3_Zone', 'perc_duration_2_Zone','perc_duration_4_Zone',
          'nbr_stays_2_Zone_perh','nbr_stays_3_Zone_perh', 'nbr_stays_5_Zone_perh','nbr_stays_4_Zone_perh',
          'SleepingHeight','vertical_travel_distance_perh',
          'in_WG_15mnAfterOpening','perc_1_Zone_while_WG_open','nbr_stays_1_Zone_perh')
#first obs per week per individual
df_pca = df[df$tobeusedPCA==1,][li_pca]
res = psych::principal(r=df_pca, rotate="none", nfactors=3, scores=TRUE, covar=FALSE) #not varimax (we want only one PC)
df_result = data.frame(predict(object=res, data=df[li_pca], old.data=df_pca))
df$PC1 = df_result$PC1 #varimax: RC, none: PC

#for validity compare with the one run on full dataset
res_validity = psych::principal(r=df[li_pca], rotate="none", nfactors=3, scores=TRUE) #varimax if we want more, none for only one
res_validity
head(df, 3)

##################################### random intercept model (RI) #######################################
#PenID & Class
#To compare models with nested fixed effects (here: class) and same random structure, ML estimation must be used and not REML
MAVG_pen_class = lmerTest::lmer(PC1 ~ time + time2 + avgDIB_scale + Treatment + CLASS + temperature_C_avg_scale + InitialWeight_scale + (1|PenID/HenID), REML=FALSE, data = df)
summary(MAVG_pen_class)
#-->penID accounted less than 1% of the variation

MAVG_pen_noclass = lmerTest::lmer(PC1 ~ time + time2 + avgDIB_scale + Treatment + temperature_C_avg_scale + InitialWeight_scale + (1|PenID/HenID), REML=FALSE, data = df)
summary(MAVG_pen_noclass)
#-->penID accounted less than 1% of the variation

anova(MAVG_pen_class, MAVG_pen_noclass, Test='Chisq')
#-->without class is better when penID is present, lets try when penID is not present

MAVG_nopen_class = lmerTest::lmer(PC1 ~ time + time2 + avgDIB_scale + Treatment + CLASS + temperature_C_avg_scale + InitialWeight_scale + (1|HenID), REML=FALSE, data = df)
summary(MAVG_nopen_class)

MAVG_nopen_noclass = lmerTest::lmer(PC1 ~ time + time2 + avgDIB_scale + Treatment + temperature_C_avg_scale + InitialWeight_scale + (1|HenID), REML=FALSE, data = df)
summary(MAVG_nopen_noclass)
r.squaredGLMM(MAVG_nopen_noclass)
###### normally distributed residuals
qqnorm(resid(MAVG_nopen_noclass))
qqline(resid(MAVG_nopen_noclass))
hist(resid(MAVG_nopen_noclass))
######check homogeneity of variance (residuals has constant variance)
plot(MAVG_nopen_noclass)

anova(MAVG_nopen_class, MAVG_nopen_noclass, Test='Chisq')
#-->without class is better when penID is not present
#-->remove penID and class, use MAVG_nopen_noclass

###### repeatability ######
set.seed(1)
simulated = sim(MAVG_nopen_noclass, n.sim = 1000)
posterior_HenID = apply(simulated@ranef$"HenID"[ , , 1],1,var)
posterior_residual  = simulated@sigma^2
quantile(posterior_HenID/(posterior_HenID+posterior_residual), prob=c(0.025, 0.5, 0.975))


########################################### random slope model ###########################################
###### Random slope 1 (RS1) ######
MPL1 = lmer(PC1 ~ time + time2 + avgDIB_scale + Treatment + temperature_C_avg_scale + InitialWeight_scale + (1 + time|HenID), 
            REML=TRUE, data=df)
summary(MPL1)
r.squaredGLMM(MPL1)
qqnorm(resid(MPL1))
qqline(resid(MPL1))
hist(resid(MPL1))
plot(MPL1)

###### Random slope 2 (RS2) ######
MPL2 = lmer(PC1 ~ time + time2 + avgDIB_scale + Treatment + temperature_C_avg_scale + InitialWeight_scale + (1 + time + time2|HenID), 
                REML=TRUE, data=df, control = lmerControl(optimizer ='optimx', optCtrl=list(method='nlminb')))
summary(MPL2)
r.squaredGLMM(MPL2)
qqnorm(resid(MPL2))
qqline(resid(MPL2))
hist(resid(MPL2))
plot(MPL2)

############################################# comparing models #############################################

#to compare with the random slope model, we will run the RI as previously selected but with RELM = TRUE, beause two models with nested
#random structures cannot be done with ML as the estimators for the variance terms are biased under ML 
MAVG_compare = lmerTest::lmer(PC1 ~ time + time2 + avgDIB_scale + Treatment + temperature_C_avg_scale + InitialWeight_scale + (1|HenID), 
                      REML=TRUE, data = df)

#because we used REML estimation, we can compare AICs
AIC(MAVG_compare, MPL1, MPL2)

########### MLP1 & MAVG_compare
#note that anova can not be used to comapre these models due to the "testing on the bounderies" issue. In other words, LRT from anova is 
#not correct anymore as we compare parameters that are on the boundery (random effects)
#p.124, Chapter 5 from: Zuur, A. F., Ieno, E. N., Walker, N., Saveliev, A. A. and Smith, G. M. 2009. Mixed Effects Models and Extensions in Ecology with R. Springer.Mixed Effects Modelling for Nested Data (https://link.springer.com/content/pdf/10.1007%2F978-0-387-87458-6_5.pdf)
L = -2*(summary(MAVG_compare)$logLik -summary(MPL1)$logLik)
pval = 0.5 * ((1 - pchisq(L, df=1)) + (1 - pchisq(L, df=2))) 
pval
#--> p-value < 0.001 --> reject H0 --> adding random slope to the model is a significant improvement

########### MLP2 & MLP1
L = -2*(summary(MPL1)$logLik -summary(MPL2)$logLik)
pval = 0.5 * ((1 - pchisq(L, df=2)) + (1 - pchisq(L, df=3)))
pval
#--> p-value < 0.001 --> reject H0 --> adding random quadratic term slope to the model is a significant improvement
#OR same result when doing (pval = 1-pchisq(L, 3)) as on p.23 of supplements of the guide provided in the paper as citation: Arnold, P. A., Kruuk, L. E. B. & Nicotra, A. B. How to analyse plant phenotypic plasticity in response to a changing climate. New Phytologist 222, 1235–1241 (2019)

######################################## predictability: Double hierarchical model ########################################
double_model = bf(PC1~ time + time2 + Treatment + temperature_C_avg_scale + (1+time+time2|a|HenID), 
                  sigma~time + time2 + Treatment + temperature_C_avg_scale + (1|a|HenID))
modelPred = brm(double_model, data=df, iter=50000, inits="random", seed=12345, control = list(max_treedepth=15), 
                cores=my.cores, chains=10, thin=3)
modelPred = add_criterion(modelPred, criterion=c('waic','bayes_R2','loo'), file=file.path(path_adapt,'BRMS_Model_variance0_pred'))
summary(modelPred)
plot(modelPred)
pp_check(modelPred, ndraws=100) #posterior predictive check
print(modelPred$criteria$loo) #approximative leave-one-out cross validation

#coefficient of variation in predictability” (CVP)
log.norm.res = exp(posterior_samples(modelPred)$"sd_HenID__sigma_Intercept"^2)
CVP = sqrt(log.norm.res-1)
mean(CVP);HPDinterval(as.mcmc(CVP),0.95)
#extract samples (draw) from the posterior distribution (posterior_samples(modelPred) depreciated) for bivariate models
df_pred = as_draws(modelPred)


######################################### Bivariate models #########################################
#df_MVT_4stat_BI.csv: csv with the predictability estimates and the first week variables for bivariate models
df = read.csv('df_MVT_4stat_BI.csv', header = TRUE, sep = ";")
df$HenID = as.factor(df$HenID)   
df$PenID = as.factor(df$PenID) 
df$cDIB = as.integer(df$DIB_startat11)
df$CLASS = as.factor(df$CLASS) 
df$Treatment = as.factor(df$Treatment)
#scale: mean-centering of the environmental variable so that intercepts reflect average values for the population and ind. 
df$temperature_C_avg_scale = scale(df$temperature_C_avg, center=TRUE, scale=TRUE)
df$time = scale(df$cDIB, center=FALSE, scale=sd(df$cDIB, na.rm = TRUE)) #as not centering, we need to add this
df$time2 = poly(df$time, degree=2, raw=TRUE)[,2]
df$avgDIB_scale = scale(df$avgDIB, center=TRUE, scale=TRUE)
df$InitialWeight_scale = scale(df$InitialWeight, center=TRUE, scale=TRUE)
#setting reference group
contrasts(df$Treatment)
df <- df %>% mutate(Treatment = relevel(Treatment, ref = "TRAN"))
contrasts(df$Treatment)
contrasts(df$CLASS)
df <- df %>% mutate(CLASS = relevel(CLASS, ref = "LEXP"))
contrasts(df$CLASS)
print(dim(df))
summary(df)
head(df,3)

nsamp = 7000
thin = 100
nitt = nsamp * thin
burn = nitt*0.15

#here done with (PC1,severity) ; same was done with (PC1, feather)
#R: 2 response, 2 residuals
#G: only G1: only one random effect: of size 4: trait (2), cdib (1) and cdib2 (1)
prior_biv = list(R = list(V = diag(c(1, 0.0001), 2, 2), nu = 1.002, fix = 2),
                 G = list(G1 = list(V = diag(1),
                                    nu = 1,
                                    alpha.mu = rep(0,1),
                                    alpha.V = diag(25^2,1,1)),
                          G2 = list(V = diag(4),
                                    nu = 4,
                                    alpha.mu = rep(0,4),
                                    alpha.V = diag(25^2,4,4))))
set.seed(123)
BV_model_S2 = MCMCglmm(cbind(PC1, severity) ~ trait-1 +
                       trait:Treatment + 
                       at.level(trait,1):time +
                       at.level(trait,1):time2 +
                       at.level(trait,1):temperature_C_avg_scale +
                       at.level(trait,2):Predictability_mean +
                       at.level(trait,2):nbr_daysnomvt_over_3days +
                       at.level(trait,2):InitialWeight_scale +
                       at.level(trait,2):CLASS,
                       random=~us(at.level(trait,2)):PenID + us(trait + at.level(trait,1):time + at.level(trait,1):time2):HenID, 
                       rcov=~idh(trait):units, 
                       family=c("gaussian","gaussian"),
                       prior=prior_biv,
                       data=df,
                       pr = TRUE, verbose = TRUE, saveX = TRUE, saveZ = TRUE,thin = thin, burnin= burn, nitt= nitt)
#random: G-structure: random effects (co)variances
#rcov R-structure : the residual (co)variances, 
#Location : fixed effects results information
plot(BV_model_S2$VCV)
#trace plots for fixed effect
plot(BV_model_S2$Sol)
summary(BV_model_S2)
autocorr.diag(BV_model_S2$Sol)
autocorr.diag(BV_model_S2$VCV)

#################################### correlations HA & random effects
###correlation between intercept and health###
corr_int_health = BV_model_S2$VCV[,"traitseverity:traitPC1.HenID"]/
  (sqrt(BV_model_S2$VCV[,"traitseverity:traitseverity.HenID"])*
     sqrt(BV_model_S2$VCV[,"traitPC1:traitPC1.HenID"]))
posterior.mode(corr_int_health)
HPDinterval(corr_int_health)

###correlation between linear slope and health###
corr_Lslope_health = BV_model_S2$VCV[,"at.level(trait, 1):time:traitseverity.HenID"]/
  (sqrt(BV_model_S2$VCV[,"traitseverity:traitseverity.HenID"])*
     sqrt(BV_model_S2$VCV[,"at.level(trait, 1):time:at.level(trait, 1):time.HenID"]))
posterior.mode(corr_Lslope_health)
HPDinterval(corr_Lslope_health)

###correlation between quadratic slope and health###
corr_Qslope_health = BV_model_S2$VCV[,"at.level(trait, 1):time2:traitseverity.HenID"]/
  (sqrt(BV_model_S2$VCV[,"traitseverity:traitseverity.HenID"])*
     sqrt(BV_model_S2$VCV[,"at.level(trait, 1):time2:at.level(trait, 1):time2.HenID"]))
posterior.mode(corr_Qslope_health)
HPDinterval(corr_Qslope_health)


#################################### check model cvg with thtree chains
prior_biv = list(R = list(V = diag(c(1, 0.0001), 2, 2), nu = 1.002, fix = 2),
                 G = list(G1 = list(V = diag(1),
                                        nu = 1,
                                        alpha.mu = rep(0,1),
                                        alpha.V = diag(25^2,1,1)),
                          G2 = list(V = diag(4),
                                        nu = 4,
                                        alpha.mu = rep(0,4),
                                        alpha.V = diag(25^2,4,4))))

chain1 = MCMCglmm(cbind(PC1, severity) ~ trait-1 +
                       trait:Treatment + 
                       at.level(trait,1):time +
                       at.level(trait,1):time2 +
                       at.level(trait,1):temperature_C_avg_scale +
                       at.level(trait,2):Predictability_mean +
                       at.level(trait,2):nbr_daysnomvt_over_3days +
                       at.level(trait,2):InitialWeight_scale +
                       at.level(trait,2):CLASS,
                       random=~us(at.level(trait,2)):PenID + us(trait + at.level(trait,1):time + at.level(trait,1):time2):HenID, 
                       rcov=~idh(trait):units, 
                       family=c("gaussian","gaussian"),
                       prior=prior_biv,
                       data=df,
                       pr = TRUE, verbose = TRUE, saveX = TRUE, saveZ = TRUE,thin = thin, burnin= burn, nitt= nitt)
summary(chain1)

chain2 = MCMCglmm(cbind(PC1, severity) ~ trait-1 +
                       trait:Treatment + 
                       at.level(trait,1):time +
                       at.level(trait,1):time2 +
                       at.level(trait,1):temperature_C_avg_scale +
                       at.level(trait,2):Predictability_mean +
                       at.level(trait,2):nbr_daysnomvt_over_3days +
                       at.level(trait,2):InitialWeight_scale +
                       at.level(trait,2):CLASS,
                       random=~us(at.level(trait,2)):PenID + us(trait + at.level(trait,1):time + at.level(trait,1):time2):HenID, 
                       rcov=~idh(trait):units, 
                       family=c("gaussian","gaussian"),
                       prior=prior_biv,
                       data=df,
                       pr = TRUE, verbose = TRUE, saveX = TRUE, saveZ = TRUE,thin = thin, burnin= burn, nitt= nitt)
summary(chain2)

chain3 = MCMCglmm(cbind(PC1, severity) ~ trait-1 +
                       trait:Treatment + 
                       at.level(trait,1):time +
                       at.level(trait,1):time2 +
                       at.level(trait,1):temperature_C_avg_scale +
                       at.level(trait,2):Predictability_mean +
                       at.level(trait,2):nbr_daysnomvt_over_3days +
                       at.level(trait,2):InitialWeight_scale +
                       at.level(trait,2):CLASS,
                       random=~us(at.level(trait,2)):PenID + us(trait + at.level(trait,1):time + at.level(trait,1):time2):HenID, 
                       rcov=~idh(trait):units, 
                       family=c("gaussian","gaussian"),
                       prior=prior_biv,
                       data=df,
                       pr = TRUE, verbose = TRUE, saveX = TRUE, saveZ = TRUE,thin = thin, burnin= burn, nitt= nitt)
summary(chain3)
combinedchains = mcmc.list(chain1$Sol, chain2$Sol, chain3$Sol) 
plot(combinedchains)
gelman.plot(combinedchains)
gelman.diag(combinedchains) # works only for a list of models


