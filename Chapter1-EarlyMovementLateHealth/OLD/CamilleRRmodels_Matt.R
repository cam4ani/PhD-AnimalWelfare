
####Some play code for looking at plasticity of one trait on another trait####
library(MCMCglmm)
###Set WD###

####load data for RI and RR models####

movementdata <- read.csv("movementdata.csv")


###prior for random intercept model###

RI_prior <- list(R = list(V = 1, nu = 1.002),
                  G = list(G1 = list(V = 1, nu = 1.002, alpha.mu = 0, alpha.V = 1000)))

###random intercept model####

nsamp <- 1500
burn <- 50000
thin <- 2000
(nitt <- burn + nsamp * thin)

RI_model <- MCMCglmm(PC1 ~ fixedeffects,
                    random = ~ HenID,
                    prior = RI_prior,
                    data = movementdat,
                    family = "gaussian",
                    pr = TRUE,
                    verbose = TRUE, 
                    saveX = TRUE, saveZ = TRUE,
                    thin = thin, burnin= burn, nitt= nitt)

summary(RI_model)

####repeatability estimates###

PCrepeat <-  RI_model$VCV[,"HenID"]/
  (RI_model$VCV[,"HenID"] + RI_model$VCV[,"units"])

posterior.mode(PCrepeat)

HPDinterval(PCrepeat)

####Random slope model####


###prior for RR model###

RR_prior <- list(R = list(V = 1, nu = 1.002),
                          G = list(G1 = list(V = diag(3), nu = 3, 
                                             alpha.mu = rep(0,3), 
                                             alpha.V = diag(3) * 25^2)))


###random regression model###


RR_model <- MCMCglmm(PC1 ~ fixedeffects,
                     random = ~ us(1 + poly(NDPT, 2, raw = TRUE)):HenID,
                                   prior = RR_prior,
                                   data = movementdat,
                                   family = "gaussian",
                                   pr = TRUE,
                                   verbose = TRUE, 
                                   saveX = TRUE, saveZ = TRUE,
                                   thin = thin, burnin = burn, nitt = nitt)


summary(RR_model)
                     
###Need to combine movement data and health data###

healthdata <- read.csv("healthdata.csv")

alldat <- rbind(movementdata, healthdata)

####Set up prior####


prior_biv <- list(R = list(V = diag(c(1, 1, 0.0001), 3, 3), nu = 0.002, fix = 2),
                  G = list(G1 = list(V = matrix(c(1,0,0,0,
                                                  0,1,0,0,
                                                  0,0,1,0,
                                                  0,0,0,1), 4, 4,
                                                byrow = TRUE),
                                        nu = 4,
                                        alpha.mu = rep(0,4),
                                        alpha.V = diag(4) * 25^2)))


BV_model <- MCMCglmm(cbind(PC1),
                     healthvariable) ~ trait-1 +
                      at.level(trait,1):fixedeffects +
                    random = ~ us(trait + poly(NDPT, 2, raw = TRUE):atlevel(trait,1)):HenID,
                    rcov = ~ idh(trait):units,
                    family = c("gaussian", "gaussion"),
                    prior = prior_biv,
                    nitt = nitt,
                    burnin = burnin,
                    thin = thin,
                    verbose = TRUE,
                    data = alldat,
                    pr = TRUE,
                    saveX = TRUE, saveZ = TRUE)

###correlation between intercept and health###

corr_int_health <- BV_model$VCV[,"traithealthvariable:traitPC1.ID"]/
  (sqrt(BV_model$VCV[,"traithealthvariable:traithealthvariable.ID"])*
     sqrt(BV_model$VCV[,"traitPC1:traitPC1.ID"]))

posterior.mode(corr_int_health)

HPDinterval(corr_int_health)

###correlation between slope and health###

corr_slope_health <- BV_model$VCV[,"poly(NDPT, 2, raw = TRUE):atlevel(trait,1):traithealthvariable.ID"]/
  (sqrt(BV_model$VCV[,"traithealthvariable:traithealthvariable.ID"])*
     sqrt(BV_model$VCV[,"poly(NDPT, 2, raw = TRUE):atlevel(trait,1):poly(NDPT, 2, raw = TRUE):atlevel(trait,1).ID"]))


posterior.mode(corr_slope_health)

HPDinterval(corr_slope_health)