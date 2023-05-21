library(lmerTest) #lmer
library(ggplot2) #ylim in plot
library(emmeans)
library(dplyr) #%>%
library(sjPlot) #tab_model


################################################
################ download data #################
################################################
#focal birds information (one row per focal bird)
path_ = 'YOUR PATH'
#create a path to save outputs
path_save_HA = file.path(path_, 'HA&Treatment')
dir.create(path_save_HA)
path_save_Egg = file.path(path_, 'Egg&Treatment')
dir.create(path_save_Egg)

#download health assessment from KBF and feather damage
df = read.csv(file.path(path_,'OFH_df_HA.csv'), header = TRUE, sep = ",")
df$HenID = as.factor(df$HenID)   
df$PenID = as.factor(df$PenID)
df$CLASS = as.factor(df$CLASS) 
df$date = as.factor(df$date) 
df$DOA = as.integer(df$DOA) 
df$Treatment = as.factor(df$Treatment)
#setting reference group
df = df %>% mutate(CLASS = relevel(CLASS, ref = "REXP"))
df = df %>% mutate(Treatment = relevel(Treatment, ref = "TRAN"))
dim(df)
df = df[!is.na(df$DOA),]
head(df,3)

#download body mass data
df_W = read.csv(file.path(path_,'OFH_df_FOCALBIRDS.csv'), header = TRUE, sep = ",")
df_W = df_W[!is.na(df_W$DOA),]
df_W$HenID = as.factor(df_W$HenID)   
df_W$PenID = as.factor(df_W$PenID)
df_W$CLASS = as.factor(df_W$CLASS) 
df_W$date = as.factor(df_W$date) 
df_W$DOA = as.integer(df_W$DOA) 
df_W$Treatment = as.factor(df_W$Treatment)
df_W = df_W %>% mutate(CLASS = relevel(CLASS, ref = "REXP"))
df_W = df_W %>% mutate(Treatment = relevel(Treatment, ref = "TRAN"))
dim(df_W)
head(df_W,3)
hist(df_W$weight)
hist(df_W$weight_norm)

#download egg-production data
df_egg = read.csv(file.path(path_,'df_eggdata.csv'), header = TRUE, sep = ",")
df_egg$PenID = factor(df_egg$PenID)
df_egg$DIB = as.integer(df_egg$DIB)
df_egg$Treatment_allpens = factor(df_egg$Treatment_allpens) #unsacaled: 0-1
#setting reference group
df = df %>% mutate(Treatment_allpens = relevel(Treatment_allpens, ref = "TRAN"))
contrasts(df$Treatment_allpens)
print(dim(df))
summary(df)
head(df,3)
#choosing first two month in the laying barn
df_egg = df_egg[(df_egg$DIB<=60)&(df_egg$DIB>=1),]


################################################
################# KBF severity #################
################################################
dim(df)
df_S = df[(!is.na(df$severity))&(df$DOA>200),]
df_S$date = factor(df_S$date) 
dim(df_S)

#penID as random effect: singular values & explainen none of the variance --> without PenID
fit_s = lmer(severity ~  CLASS + date+Treatment+date:Treatment + (1|HenID), data=df_S)
summary(fit_s)
anova(fit_s)
######normally distributed residuals
qqnorm(resid(fit_s))
qqline(resid(fit_s))
hist(resid(fit_s))
######check homogeneity of variance (residuals has constant variance)
plot(fit_s)

#compare with and without interaction to assess significance of the predictor
fit_s0 = lmer(severity ~  CLASS + date+Treatment + (1|HenID), data=df_S)
anova(fit_s, fit_s0)
anova(fit_s0)

################################################
################ Feather damage ################
################################################
dim(df)
df_F = df[(!is.na(df$FeatherDamage))&(df$DOA>230),]
df_F$date = factor(df_F$date) 
dim(df_F)

#penID as random effect: singular values -->withouPenID
fit_f = lmer(FeatherDamage ~  CLASS + date+Treatment+date:Treatment + (1|PenID/HenID), data=df_F)
summary(fit_f)
anova(fit_f)
write.csv(anova(fit_f), file=file.path(path_save_HA, paste0('OFH_FD_anova.csv')) )
######normally distributed residuals
qqnorm(resid(fit_f))
qqline(resid(fit_f))
hist(resid(fit_f))
######check homogeneity of variance (residuals has constant variance)
plot(fit_f)

#compare with and without interaction to assess significance of the predictor
fit_f0 = lmer(FeatherDamage ~  CLASS + date+Treatment + (1|PenID/HenID), data=df_F)
anova(fit_f, fit_f0)

###############################################
################## Body mass ##################
###############################################
#penID as random effect: singular values -->withouPenID
fit_w = lmer(weight_norm ~ CLASS + date+Treatment+date:Treatment + (1|HenID), data=df_W)
summary(fit_w)
anova(fit_w)
write.csv(anova(fit_w), file=file.path(path_save_HA, paste0('OFH_Weight_anova.csv')) )
######normally distributed residuals
qqnorm(resid(fit_w))
qqline(resid(fit_w))
hist(resid(fit_w))
######check homogeneity of variance (residuals has constant variance)
plot(fit_w)

#compare with and without interaction to assess significance of the predictor
fit_w0 = lmer(weight_norm ~ CLASS + date+Treatment + (1|HenID), data=df_W)
anova(fit_w, fit_w0)


###############################################
############# Egg sigmoid curve ###############
###############################################
li = list()
tiff(file.path(path_save_Egg, 'egg_growth_crv.tiff'), width=400, height=400)
df_OFH = df_egg[(df_egg$Treatment_allpens=='OFH'),]
df_TRAN = df_egg[(df_egg$Treatment_allpens=='TRAN'),]
plot(df_OFH$DIB, df_OFH$X.eggPerTier, col=rgb(red = 0, green = 0, blue = 1, alpha = 0.1))
points(df_TRAN$DIB, df_TRAN$X.eggPerTier, col=rgb(red = 1, green = 0, blue = 0, alpha = 0.1))
i = 1
for (penid in unique(df$PenID)){
    print('----------------------------------------------')
    print(penid)
    df_penid = df_egg[(df_egg$PenID==penid),]
    df_penid = arrange(df_penid, DIB, by_group = FALSE)
    color = rgb(red = 0, green = 0, blue = 1, alpha = 0.6)
    if (unique(df_penid$Treatment_allpens)[[1]]=='OFH'){color = rgb(red = 1, green = 0, blue = 0, alpha = 0.6)}
    head(df_penid,3)
    fit_penid = nls(X.eggPerTier ~ SSlogis(DIB, Asym, xmid, scal), data = df_penid) #SSlogis() logistic growth curve
    #Asym, xmid, scal are param that will be estiamted, you can name them as you want
    #from documentation (https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/SSlogis) :
    #Asym: a numeric parameter representing the asymptote (i.e. max posisble value)
    #xmid: a numeric param. representing the x value at the inflection pt of the crv. The value of SSlogis will be Asym/2 at xmid
    #scal: a numeric scale parameter on the input axis.
    print(summary(fit_penid))
    lines(df_penid$DIB, predict(fit_penid), col=color, lwd=1.2)  
    df_res = as.data.frame(summary(fit_penid)$coefficients)
    df_res$PenID = penid
    li[[i]] = df_res
    i = i + 1
}

dev.off()
df_res = rbind(li[[1]],li[[2]],li[[3]],li[[4]],li[[5]],li[[6]],li[[7]],li[[8]],li[[9]],li[[10]])
write.csv(df_res, file.path(path_save_Egg,'egg_growthcurve.csv')) 
df_res

