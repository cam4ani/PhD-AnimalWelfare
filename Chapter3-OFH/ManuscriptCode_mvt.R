library(dplyr) #%>%
library(MuMIn) #r2
library(ggplot2) #ylim in plot
library(ggeffects)  #ggpredict
library(ggpubr) #ggboxplot
library(DHARMa) #sim
library(glmmTMB) 
# plot_summs 
library(jtools) 
library(ggstance)
library(arm) #rescale by 2sd to comapre bin with cont
library(sjPlot) #plot_models, tab_model
library(parameters) #bootstrapped p-val



################################################
################ download data #################
################################################
path_ = 'YOUR PATH'
path_visual = file.path(path_, 'Mvt&Treatment')
dir.create(path_visual)
df = read.csv(file.path(path_,'OFH_df_all_mvt.csv'), header = TRUE, sep = ",")

#factors
df$HenID = as.factor(df$HenID)   
df$PenID = as.factor(df$PenID)
df$CLASS = as.factor(df$CLASS) 
df = df %>% mutate(CLASS = relevel(CLASS, ref = "REXP"))
contrasts(df$CLASS)
df$Treatment = as.factor(df$Treatment) #unscaled: 0-1
#setting reference group
df <- df %>% mutate(Treatment = relevel(Treatment, ref = "TRAN"))
contrasts(df$Treatment)
df$monthILB = as.factor(df$monthILB) 
dim(df)
df = subset(df, !is.na(KBF_interp))
dim(df)
df = subset(df, !is.na(weight_interp))
dim(df)

#set the number of iteration for the bootstrap
n_iter = 500


#rename so that models fit into one line, for readibility
df = df %>% rename(P2Zone = propindoor_duration_2_Zone, P5Zone=propindoor_duration_5_Zone_rs, Mid4Z=mid_cum_Z4_h_Morning,
                   VTD=vertical_travel_distance, P4Zone=propindoor_duration_4_Zone_rs)


#Note: we will fit models for each movements separately, as they have different "constraint", and  aloop is here not appropriate

################################################
########## propindoor_duration_2_Zone ##########
################################################
var = 'P2Zone'
path_visual_ = file.path(path_visual, var)
dir.create(path_visual_)
dim(df)
df_ = df
dim(df_)

li_model = list()
for (i in 1:10){
    monthILBID = paste0('month',i)
    print(monthILBID)
    df__ = df_[df_$monthILB==monthILBID,]
    #drop factors not existing in this subdataframe
    df__$HenID = factor(df__$HenID) 
    df__$PenID = factor(df__$PenID)
    df__$CLASS = factor(df__$CLASS) 
    #monthILB1-monthILB2: no KBF but nbr_h_per_day_sc, after KBF but no nbr_h_per_day_sc
    if (monthILBID %in% c('month1')){
        fit_nopenid = glmmTMB(went_litter ~  CLASS+ Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(nbr_h_per_day) + (1|HenID),
                              data=df__, family=binomial)
        fit_penid = glmmTMB(went_litter ~  CLASS+ Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(nbr_h_per_day)+(1|PenID/HenID), 
                            data=df__, family=binomial)}
    else if (monthILBID %in% c('month2')){
        fit_nopenid = glmmTMB(P2Zone ~  CLASS+ Treatment + rescale(DIB) + rescale(weight_interp)+rescale(nbr_h_per_day) + (1|HenID),
                              data=df__, family=gaussian)
        fit_penid = glmmTMB(P2Zone ~  CLASS+ Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(nbr_h_per_day) + (1|PenID/HenID),
                            data=df__, family=gaussian)}
    else{
        fit_nopenid = glmmTMB(P2Zone ~  CLASS + Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(KBF_interp) + (1|HenID),
                              data=df__, family=gaussian)
        fit_penid = glmmTMB(P2Zone ~  CLASS + Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(KBF_interp) + (1|PenID/HenID),
                            data=df__, family=gaussian)}
    set.seed(0)

    #test if PenID should be kept or not
    print(anova(fit_nopenid,fit_penid))
    p_value = anova(fit_nopenid,fit_penid)['Pr(>Chisq)']['fit_penid',]
    if (p_value<0.05){
        fit = fit_penid    
    }else{
        fit = fit_nopenid    
    } 
        
    #binomial: exponentiate
    if (monthILBID %in% c('month1')){
        df_res = model_parameters(fit, bootstrap = TRUE, iterations = n_iter,effects = "all",exponentiate = TRUE)
        write.csv(df_res, file=file.path(path_visual_, paste0(var, 'bootstrapPVAL',monthILBID,'.csv')) )
    }
    
    #gaussian model additional (to darmha) residuals inspection & exponentiate
    else{
        tiff(file.path(path_visual_, paste0(var,'_qqplot',monthILBID,'.tiff')), width=400, height=400)
        qqnorm(resid(fit))
        qqline(resid(fit))
        dev.off()

        tiff(file.path(path_visual_, paste0(var,'_reshisto',monthILBID,'.tiff')), width=400, height=400)
        hist(resid(fit))
        dev.off()
        ######check homogeneity of variance (residuals has constant variance)
        #into with glmmTMB but we used it with glmer and it was all food. for glmmtmb we will use darmha
        #tiff(file.path(path_visual_, paste0(var,'_homogeneityvar',monthILBID,'.tiff')), width=400, height=400)
        #print(plot(fit))
        #dev.off()        
        df_res = model_parameters(fit, bootstrap = TRUE, iterations = n_iter,effects = "all",exponentiate = FALSE)
        write.csv(df_res, file=file.path(path_visual_, paste0(var, 'bootstrapPVAL',monthILBID,'.csv')) )
        
    }

    li_model[[i]] = fit
    
    #save raw coeff in case
    write.csv(summary(fit)[['coefficients']]$cond, file=file.path(path_visual_, paste0(var,monthILBID,'.csv')) )
    
    simulationOutput = simulateResiduals(fittedModel=fit)
    tiff(file.path(path_visual_, paste0(var,'_SimRes1',monthILBID,'.tiff')), width=400, height=400)
    print(plot(simulationOutput))
    dev.off()
    
    tiff(file.path(path_visual_, paste0(var,'_SimRes2',monthILBID,'.tiff')), width=400, height=400)
    print(plotResiduals(simulationOutput, form = df__$Treatment))
    dev.off()
    
    tab_model(fit, collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel',monthILBID,'.doc')))
}

#save ta model in case
tab_model(li_model[1:5], collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel1-5.doc')))
tab_model(li_model[6:10], collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel6-10.doc')))

#save visuals in case
tiff(file.path(path_visual_, paste0(var,'_OFHCoeff_month1','.tiff')), width=400, height=400)
plot_summs(li_model[1], coefs=c('TreatmentOFH'), colors='Rainbow', legend.title=var, exp=TRUE)
dev.off()

#to match the colors and modelID, put as model 1 the model2. then, we will remove it from the plot
li_model_ = li_model
li_model_[[1]] = li_model_[[2]]
tiff(file.path(path_visual_, paste0(var,'_OFHCoeff_month2-10','.tiff')), width=400, height=400)
plot_summs(li_model_, coefs=c('TreatmentOFH'), colors='Rainbow', legend.title=var, exp=FALSE)
dev.off()





################################################
######## propindoor_duration_5_Zone_rs #########
################################################
var = 'P5Zone'
path_visual_ = file.path(path_visual, var)
dir.create(path_visual_)
dim(df)
df_ = df
dim(df_)

li_model = list()
for (i in 1:10){
    monthILBID = paste0('month',i)
    print(monthILBID)
    df__ = df_[df_$monthILB==monthILBID,]
    #drop factors not existing in this subdataframe
    df__$HenID = factor(df__$HenID) 
    df__$PenID = factor(df__$PenID)
    df__$CLASS = factor(df__$CLASS) 
    #monthILB1-monthILB2: no KBF but nbr_h_per_day_sc, after KBF but no nbr_h_per_day_sc
    if (monthILBID %in% c('month1', 'month2')){
        fit_nopenid = glmmTMB(P5Zone ~  CLASS+ Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(nbr_h_per_day)+(1|HenID), 
                   data=df__, family=beta_family(link="logit"))
        fit_penid = glmmTMB(P5Zone ~  CLASS+ Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(nbr_h_per_day)+(1|PenID/HenID), 
                   data=df__, family=beta_family(link="logit"))}
    else{
        fit_nopenid = glmmTMB(P5Zone ~  CLASS + Treatment + rescale(DIB) + rescale(weight_interp)+rescale(KBF_interp) + (1|HenID), 
                   data=df__, family=beta_family(link="logit"))
        fit_penid = glmmTMB(P5Zone ~  CLASS + Treatment + rescale(DIB) + rescale(weight_interp)+rescale(KBF_interp) + (1|PenID/HenID), 
                   data=df__, family=beta_family(link="logit"))}
    print(anova(fit_nopenid,fit_penid))
    p_value = anova(fit_nopenid,fit_penid)['Pr(>Chisq)']['fit_penid',]
    if (p_value<0.05){
        fit = fit_penid    
    }else{
        fit = fit_nopenid    
    }    
    
    li_model[[i]] = fit
    write.csv(summary(fit)['coefficients']$coefficients$cond, file=file.path(path_visual_, paste0(var,monthILBID,'.csv')) )
    
    set.seed(0)
    
    df_res = model_parameters(fit, bootstrap = TRUE, iterations = n_iter,effects = "all",exponentiate = TRUE)
    write.csv(df_res, file=file.path(path_visual_, paste0(var, 'bootstrapPVAL',monthILBID,'.csv')) )

    simulationOutput = simulateResiduals(fittedModel=fit)
    tiff(file.path(path_visual_, paste0(var,'_SimRes1',monthILBID,'.tiff')), width=400, height=400)
    print(plot(simulationOutput))
    dev.off()
    
    tiff(file.path(path_visual_, paste0(var,'_SimRes2',monthILBID,'.tiff')), width=400, height=400)
    print(plotResiduals(simulationOutput, form = df__$Treatment))
    dev.off()
    tab_model(fit, collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel',monthILBID,'.doc')))
}


#tiff(file.path(path_visual_, paste0(var,'_OFHCoeff','.tiff')), width=400, height=400)
#plot_summs(li_model, coefs=c('TreatmentOFH'), colors='Rainbow', legend.title=var, exp=TRUE)
#dev.off()
#tab_model(li_model[1:5], collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel1-5.doc')))
#tab_model(li_model[6:10], collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel6-10.doc')))



################################################
############# mid_cum_Z4_h_Morning #############
################################################
var = 'Mid4Z'
path_visual_ = file.path(path_visual, var)
dir.create(path_visual_)
dim(df)
df_ = subset(df, !is.na(Mid4Z))
dim(df_)

li_model = list()
for (i in 2:10){
    monthILBID = paste0('month',i)
    print(monthILBID)
    df__ = df_[df_$monthILB==monthILBID,]
    #drop factors not existing in this subdataframe
    df__$HenID = factor(df__$HenID) 
    df__$PenID = factor(df__$PenID)
    df__$CLASS = factor(df__$CLASS) 
    #monthILB1-monthILB2: no KBF but nbr_h_per_day_sc, after KBF but no nbr_h_per_day_sc
    if (monthILBID %in% c('month1', 'month2')){
        fit_nopenid = glmmTMB(Mid4Z ~  CLASS+ Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(nbr_h_per_day)+(1|HenID), 
                   data=df__, family=Gamma(link = "log"))
        fit_penid = glmmTMB(Mid4Z ~  CLASS+ Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(nbr_h_per_day)+(1|PenID/HenID), 
                   data=df__, family=Gamma(link = "log"))}
    else{
        fit_nopenid = glmmTMB(Mid4Z ~  CLASS + Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(KBF_interp) + (1|HenID), 
                   data=df__, family=Gamma(link = "log"))
        fit_penid = glmmTMB(Mid4Z ~  CLASS + Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(KBF_interp) + (1|PenID/HenID), 
                   data=df__, family=Gamma(link = "log"))}
    print(anova(fit_nopenid,fit_penid))
    p_value = anova(fit_nopenid,fit_penid)['Pr(>Chisq)']['fit_penid',]
    if (p_value<0.05){
        fit = fit_penid    
    }else{
        fit = fit_nopenid    
    }    
    
    li_model[[i]] = fit
    write.csv(summary(fit)['coefficients']$coefficients$cond, file=file.path(path_visual_, paste0(var,monthILBID,'.csv')) )
    
    set.seed(0)
    
    df_res = model_parameters(fit, bootstrap = TRUE, iterations = n_iter,effects = "all",exponentiate = TRUE)
    write.csv(df_res, file=file.path(path_visual_, paste0(var, 'bootstrapPVAL',monthILBID,'.csv')) )

    simulationOutput = simulateResiduals(fittedModel=fit)
    tiff(file.path(path_visual_, paste0(var,'_SimRes1',monthILBID,'.tiff')), width=400, height=400)
    print(plot(simulationOutput))
    dev.off()
    
    tiff(file.path(path_visual_, paste0(var,'_SimRes2',monthILBID,'.tiff')), width=400, height=400)
    print(plotResiduals(simulationOutput, form = df__$Treatment))
    dev.off()
    tab_model(fit, collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel',monthILBID,'.doc')))
}


#to match the colors and modelID, put as model 1 the model2. then, we will remove it
li_model[[1]] = li_model[[2]]

tiff(file.path(path_visual_, paste0(var,'_OFHCoeff','.tiff')), width=400, height=400)
plot_summs(li_model[1:10], coefs=c('TreatmentOFH'), colors='Rainbow', legend.title=var, exp=TRUE)
dev.off()
tab_model(li_model[2:5], collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel2-5.doc')))
tab_model(li_model[6:10], collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel6-10.doc')))





################################################
########### vertical_travel_distance ###########
################################################
var = 'VTD'
path_visual_ = file.path(path_visual, var)
dir.create(path_visual_)
dim(df)
df_ = df
dim(df_)
#month: poisson - exponentate, other months: gaussian - dont exponentiate
li_model = list()
for (i in 1:10){
    monthILBID = paste0('month',i)
    print(monthILBID)
    df__ = df_[df_$monthILB==monthILBID,]
    #drop factors not existing in this subdataframe
    df__$HenID = factor(df__$HenID) 
    df__$PenID = factor(df__$PenID)
    df__$CLASS = factor(df__$CLASS) 
    #monthILB1-monthILB2: no KBF but nbr_h_per_day_sc, after KBF but no nbr_h_per_day_sc
    if (monthILBID %in% c('month1')){
        fit_nopenid = glmmTMB(VTD ~  CLASS+ Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(nbr_h_per_day)+(1|HenID), 
                   data=df__, family=poisson, zi=~1+rescale(DIB))
        fit_penid = glmmTMB(VTD ~  CLASS+ Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(nbr_h_per_day)+(1|PenID/HenID), 
                   data=df__, family=poisson, zi=~1+rescale(DIB))}
    else if (monthILBID %in% c('month2')){
        fit_nopenid = glmmTMB(VTD ~  CLASS+ Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(nbr_h_per_day)+(1|HenID), 
                   data=df__, family=gaussian)
        fit_penid = glmmTMB(VTD ~  CLASS+ Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(nbr_h_per_day)+(1|PenID/HenID), 
                   data=df__, family=gaussian)}
    else{
        fit_nopenid = glmmTMB(VTD ~  CLASS + Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(KBF_interp) + (1|HenID), 
                   data=df__, family=gaussian)
        fit_penid = glmmTMB(VTD ~  CLASS + Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(KBF_interp) + (1|PenID/HenID), 
                   data=df__, family=gaussian)}    
    print(anova(fit_nopenid,fit_penid))
    p_value = anova(fit_nopenid,fit_penid)['Pr(>Chisq)']['fit_penid',]
    if (p_value<0.05){
        fit = fit_penid    
    }else{
        fit = fit_nopenid    
    }
    
    li_model[[i]] = fit
    write.csv(summary(fit)[['coefficients']]$cond, file=file.path(path_visual_, paste0(var,monthILBID,'.csv')) )
    
    set.seed(0)
    if (monthILBID %in% c('month1')){
        df_res = model_parameters(fit, bootstrap = TRUE, iterations = n_iter,effects = "all",exponentiate = TRUE)
        write.csv(df_res, file=file.path(path_visual_, paste0(var, 'bootstrapPVAL',monthILBID,'.csv')) )}
    else{
        df_res = model_parameters(fit, bootstrap = TRUE, iterations = n_iter,effects = "all",exponentiate = FALSE)
        write.csv(df_res, file=file.path(path_visual_, paste0(var, 'bootstrapPVAL',monthILBID,'.csv')) )
    }
    simulationOutput = simulateResiduals(fittedModel=fit)
    tiff(file.path(path_visual_, paste0(var,'_SimRes1',monthILBID,'.tiff')), width=400, height=400)
    print(plot(simulationOutput))
    dev.off()
    
    tiff(file.path(path_visual_, paste0(var,'_SimRes2',monthILBID,'.tiff')), width=400, height=400)
    print(plotResiduals(simulationOutput, form = df__$Treatment))
    dev.off()
    tab_model(fit, collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel',monthILBID,'.doc')))
}


#save in case
tab_model(li_model[1:5], collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel1-5.doc')))
tab_model(li_model[6:10], collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel6-10.doc')))

tiff(file.path(path_visual_, paste0(var,'_OFHCoeff_month1','.tiff')), width=400, height=400)
plot_summs(li_model[1], coefs=c('TreatmentOFH'), colors='Rainbow', legend.title=var, exp=TRUE)
dev.off()

#to match the colors and modelID, put as model 1 the model2. then, we will remove it from the plot
li_model_ = li_model
li_model_[[1]] = li_model_[[2]]
tiff(file.path(path_visual_, paste0(var,'_OFHCoeff_month2-10','.tiff')), width=400, height=400)
plot_summs(li_model_, coefs=c('TreatmentOFH'), colors='Rainbow', legend.title=var, exp=FALSE)
dev.off()


################################################
######## propindoor_duration_4_Zone_rs #########
################################################
var = 'P4Zone'
path_visual_ = file.path(path_visual, var)
dir.create(path_visual_)
dim(df)
#as until then they only use it as a transitional zone
df_ = subset(df, !is.na(P4Zone))
dim(df_)
df_ = df_[df_['HenID']!='hen_199', ]
dim(df_)
li_model = list()
for (i in 2:10){
    monthILBID = paste0('month',i)
    print(monthILBID)
    df__ = df_[df_$monthILB==monthILBID,]
    print(dim(df__))
    #drop factors not existing in this subdataframe
    df__$HenID = factor(df__$HenID) 
    df__$PenID = factor(df__$PenID)
    df__$CLASS = factor(df__$CLASS) 
    #monthILB1-monthILB2: no KBF but nbr_h_per_day_sc, after KBF but no nbr_h_per_day_sc
    if (monthILBID %in% c('month1', 'month2')){
        fit_nopenid = glmmTMB(P4Zone ~  CLASS+ Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(nbr_h_per_day)+(1|HenID), 
                   data=df__, family=beta_family(link="logit"))
        fit_penid = glmmTMB(P4Zone ~  CLASS+ Treatment + rescale(DIB) + rescale(weight_interp)+ rescale(nbr_h_per_day)+(1|PenID/HenID), 
                   data=df__, family=beta_family(link="logit"))}
    else{
        fit_nopenid = glmmTMB(P4Zone ~  CLASS + Treatment + rescale(DIB) +rescale(weight_interp)+ rescale(KBF_interp) + (1|HenID), 
                   data=df__, family=beta_family(link="logit"))
        fit_penid = glmmTMB(P4Zone ~  CLASS + Treatment + rescale(DIB) +rescale(weight_interp)+ rescale(KBF_interp) + (1|PenID/HenID), 
                   data=df__, family=beta_family(link="logit"))}
    print(anova(fit_nopenid,fit_penid))
    p_value = anova(fit_nopenid,fit_penid)['Pr(>Chisq)']['fit_penid',]
    if (p_value<0.05){
        fit = fit_penid    
    }else{
        fit = fit_nopenid    
    }
    
    li_model[[i]] = fit
    write.csv(summary(fit)['coefficients']$coefficients$cond, file=file.path(path_visual_, paste0(var,monthILBID,'.csv')) )
    
    set.seed(0)
        
    df_res = model_parameters(fit, bootstrap = TRUE, iterations = n_iter,effects = "all",exponentiate = TRUE)
    write.csv(df_res, file=file.path(path_visual_, paste0(var, 'bootstrapPVAL',monthILBID,'.csv')) )

    simulationOutput = simulateResiduals(fittedModel=fit)
    tiff(file.path(path_visual_, paste0(var,'_SimRes1',monthILBID,'.tiff')), width=400, height=400)
    print(plot(simulationOutput))
    dev.off()
    
    tiff(file.path(path_visual_, paste0(var,'_SimRes2',monthILBID,'.tiff')), width=400, height=400)
    print(plotResiduals(simulationOutput, form = df__$Treatment))
    dev.off()
    tab_model(fit, collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel',monthILBID,'.doc')))

}

#to match the colors and modelID, put as model 1 the model2. then, we will remove it
li_model[[1]] = li_model[[2]]

tiff(file.path(path_visual_, paste0(var,'_OFHCoeff','.tiff')), width=400, height=400)
plot_summs(li_model[1:10], coefs=c('TreatmentOFH'), colors='Rainbow', legend.title=var, exp=TRUE)
dev.off()
tab_model(li_model[2:5], collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel1-5.doc')))
tab_model(li_model[6:10], collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel6-10.doc')))




################################################
################### WentinWG ###################
################################################
var = 'WentinWG'
path_visual_ = file.path(path_visual, var)
dir.create(path_visual_)
dim(df)
df_ = subset(df, !is.na(WentinWG))
dim(df_)
df_ = subset(df_, !is.na(temperature_C_avg))
dim(df_)

li_model = list()
for (i in 1:10){
    monthILBID = paste0('month',i)
    print(monthILBID)
    df__ = df_[df_$monthILB==monthILBID,]
    #drop factors not existing in this subdataframe
    df__$HenID = factor(df__$HenID) 
    df__$PenID = factor(df__$PenID)
    df__$CLASS = factor(df__$CLASS) 
    #monthILB1-monthILB2: no KBF but nbr_h_per_day_sc, after KBF but no nbr_h_per_day_sc
    if (monthILBID %in% c('month1', 'month2')){
        fit_penid = glmmTMB(WentinWG ~  CLASS+ Treatment + rescale(DIB) +rescale(weight_interp)+rescale(nbr_h_per_day)+ rescale(nbr_h_WGopen)+ rescale(temperature_C_avg)+(1|PenID/HenID), 
                   data=df__, family=binomial)
        
        fit_nopenid = glmmTMB(WentinWG ~  CLASS+ Treatment + rescale(DIB) +rescale(weight_interp)+rescale(nbr_h_per_day)+ rescale(nbr_h_WGopen)+ rescale(temperature_C_avg)+(1|HenID), 
                   data=df__, family=binomial)}
    
    else{
        fit_penid = glmmTMB(WentinWG ~  CLASS + Treatment + rescale(DIB) +rescale(weight_interp)+ rescale(KBF_interp) +rescale(nbr_h_WGopen)+ rescale(temperature_C_avg) + (1|PenID/HenID), 
                   data=df__, family=binomial)
        
        fit_nopenid = glmmTMB(WentinWG ~  CLASS + Treatment + rescale(DIB) +rescale(weight_interp)+ rescale(KBF_interp) +rescale(nbr_h_WGopen)+ rescale(temperature_C_avg) + (1|HenID), 
                   data=df__, family=binomial)} 
    
    print(anova(fit_nopenid,fit_penid))
    p_value = anova(fit_nopenid,fit_penid)['Pr(>Chisq)']['fit_penid',]
    if (p_value<0.05){
        fit = fit_penid    
    }else{
        fit = fit_nopenid    
    }

    li_model[[i]] = fit
    write.csv(summary(fit)['coefficients']$coefficients$cond, file=file.path(path_visual_, paste0(var,monthILBID,'.csv')) )

    set.seed(0)
    
    df_res = model_parameters(fit, bootstrap = TRUE, iterations = n_iter,effects = "all",exponentiate = TRUE)
    write.csv(df_res, file=file.path(path_visual_, paste0(var, 'bootstrapPVAL',monthILBID,'.csv')) )

    simulationOutput = simulateResiduals(fittedModel=fit)
    tiff(file.path(path_visual_, paste0(var,'_SimRes1',monthILBID,'.tiff')), width=400, height=400)
    print(plot(simulationOutput))
    dev.off()
    
    tiff(file.path(path_visual_, paste0(var,'_SimRes2',monthILBID,'.tiff')), width=400, height=400)
    print(plotResiduals(simulationOutput, form = df__$Treatment))
    dev.off()
    tab_model(fit, collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel',monthILBID,'.doc')))
}

tiff(file.path(path_visual_, paste0(var,'_OFHCoeff','.tiff')), width=400, height=400)
plot_summs(li_model, coefs=c('TreatmentOFH'), colors='Rainbow', legend.title=var, exp=TRUE)
dev.off()
#truncated_nbinom2

tab_model(li_model[1:5], collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel1-5.doc')))
tab_model(li_model[6:10], collapse.ci = TRUE, p.style = "numeric_stars", file=file.path(path_visual_, paste0(var,'_TableModel6-10.doc')))









