library(ctsem)
library(dplyr)
library(plyr)
libraary(ICC)
library(Rcpp)

setwd("G:/VPHI/Welfare/2- Research Projects/Camille Montalcini/Origins.GS/DataOutput/TrackingSystem/EXP2_")

################# download data ############ 
path_save = file.path('LongformatModels')
dir.create(path_save)
df = read.csv('df_longformatALL.csv', header = TRUE, sep = ",")
df$HenID = as.factor(df$HenID)   
df$ExperimentID = as.factor(df$ExperimentID)   
#tme is for now week of age
summary(df)
print(dim(df))
head(df)
#describe(df)

#TODO on final model, only relevant when optimize=FALSE, 
iter_ = 2000 
chains_ = 2 
bool_optimize = TRUE #documentation FALSE: "which also mean intoverpop will be set to FALSE automatically and therefore we integrates over 
#full sampling rather than population distribution of parameters"
#number of cpu cores to use to speed up
cores_ = 5 
folds_ = 3


################# prepare data ############ 
#initialise what we want
name = 'VTD_severity_trend'
data_mvtha = df %>% select (id, MonthInstudy, severity, VTDperhour, time2event,IsOFH,Is_exp2,Is_exp3,weight)#, IsOFH, Is_relocated,Is_exp2,Is_exp3)
data_mvtha = plyr::rename(data_mvtha, c('MonthInstudy'='time',"severity"="Y1", "VTDperhour"="Y2"))

#create directory
path_save_ = file.path(path_save,name)
dir.create(path_save_)

#centering and scaling variables in the model
li_sc = c('Y1','Y2')
data_mvtha[,li_sc] = scale(data_mvtha[,li_sc]) 
#warning if we have double and integer, and as Y1 and Y2 are double, let set them all
data_mvtha[li_sc] = sapply(data_mvtha[li_sc],as.double)
data_mvtha[c('IsOFH','Is_exp2','Is_exp3')] = sapply(data_mvtha[c('IsOFH','Is_exp2','Is_exp3')],as.double)

dim(data_mvtha)
head(data_mvtha,2)
summary(data_mvtha)
str(data_mvtha)


################# Model ############ 
Mdiffu = matrix(c(0,0,0,0,0,0,
              0,0,0,0,0,0,
              0,0,0,0,0,0,
              0,0,0,0,0,0,
              0,0,0,0,'diffu_dynKBF','diffu_dynVTD_dynKBF',
              0,0,0,0,0,'diffu_dynVTD'), ncol = 6)
Mdiffu


#"initial covariance matrices are fixed to zero in many elements such that these new dynamics processes do not interact or 
#co-vary with the other processes in the system"
Mt0var = matrix(c('T0var_KBF','T0var_VTD_KBF','T0var_cintKBF_KBF','T0var_cintVTD_KBF',0,0,
                  0,'T0var_VTD','T0var_cintKBF_VTD','T0var_cintVTD_VTD',0,0,
                  0,0,'T0var_cintKBF','T0var_cintVTD_cintKBF',0,0,
                  0,0,0,'T0var_cintVTD',0,0,
                  0,0,0,0,0,0,
                  0,0,0,0,0,0), ncol=6)
Mt0var

#The initial state and continuous intercept for each dynamical process is set to zero, to ensure that the processes capture 
#only the fluctuations and not the general trends
Mt0means = matrix(c('T0m_KBF','T0m_VTD','T0m_cintKBF','T0m_cintVTD',0,0),ncol=1)
Mt0means
MCINT = matrix(c(0,0,0,0,0,0),ncol=1)


#"The auto-effect of the dynamics processes is estimated, setting the speed of the fluctuations"
Mdrift = matrix(c('drift_KBF',0,0,0,0,0,
                    0,'drift_VTD',0,0,0,0,
                    1,0,0,0,0,0,
                    0,1,0,0,0,0,
                    0,0,0,0,'drift_dynKBF','drift_dynVTD_dynKBF',
                    0,0,0,0,'drift_dynKBF_dynVTD','drift_dynVTD'), ncol = 6)
Mdrift


#initialize the model with two latent each with one manifest variable
model_fit = ctModel(type='stanct', n.manifest=2, n.latent=6, id="id", time="time",
                    manifestNames=c('Y1','Y2'), latentNames=c('KBF','VTD','cintKBF','cintVTD','dynKBF','dynVTD'), 
                    T0MEANS=Mt0means, #initial state of dynamical process set to 0
                    T0VAR=Mt0var,
                    CINT = MCINT,
                    MANIFESTMEANS=matrix(c(0,0),ncol=1),
                    n.TIpred = 1, TIpredNames=c('Is_exp3'),
                    LAMBDA=matrix(c(1,0, 
                                    0,1,
                                    0,0,
                                    0,0,
                                    1,0,
                                    0,1), ncol=6),  
                    DRIFT=Mdrift,
                    tipredDefault=FALSE,
                    DIFFUSION=Mdiffu) 

#we let ind. var in trend
model_fit$pars$indvarying[model_fit$pars$matrix %in% c('T0MEANS','CINT')] = TRUE
model_fit$pars$Is_exp3_effect[(model_fit$pars$matrix %in% c('CINT','T0MEANS'))&(model_fit$pars$row %in% c(1,3))] = TRUE
model_fit$pars
#check what we did
model_fit$pars[model_fit$pars$matrix %in% c('T0MEANS','CINT'),]


set.seed(0)
fit_mvtha = ctStanFit(datalong=data_mvtha, ctstanmodel=model_fit, verbose=0, cores=cores_, nopriors=FALSE,
                      iter=iter_, chains=chains_, optimize=bool_optimize)
ctsem:::ctSummarise(fit_mvtha, folder=file.path(path_save_,name), cores=cores_, ctStanPlotPost=F, nsamples=1000)








