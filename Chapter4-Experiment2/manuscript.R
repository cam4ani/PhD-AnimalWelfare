library(ctsem)
library(dplyr)
library(plyr)

############################## download data ##############################
setwd("YOUR ONE")
path_data = file.path('LongformatModels') #folder name in the directory
dir.create(path_data)
df = read.csv('df_longformatALL.csv', header = TRUE, sep = ",")
df$HenID = as.factor(df$HenID)   
#df$id = as.factor(df$id)    #otherwise original and new ID differ
df$ExperimentID = as.factor(df$ExperimentID) 
summary(df)
print(dim(df))
head(df)


#number of cpu cores to use to speed up
cores_ = 10 

#initialise the saptial behaviour we want among: VTDpaerhour PropZ5 zonecrossed2transition unevenness
variable = 'VTDperhour' 
name = paste0(variable, '_KBF_trend_rs0FinalciBHV_')
#create directory where to save output
path_save_ = file.path(path_data,name)
dir.create(path_save_)


############################## data processing ##############################
data_mvtha = df[,c('id', 'MonthInstudy', 'severity', variable,'IsOFH','Dataset1','Dataset3','Is_relocated')]
#centering and scaling variables in the model
li_sc = c('severity',variable)
data_mvtha[,li_sc] = scale(data_mvtha[,li_sc]) 
#warning if we have double and integer, and as Y1 and Y2 are double, let set them all
data_mvtha[li_sc] = sapply(data_mvtha[li_sc],as.double)
data_mvtha[c('IsOFH','Dataset1','Dataset3','Is_relocated')] = sapply(data_mvtha[c('IsOFH','Dataset1','Dataset3','Is_relocated')],as.double)
#show info
dim(data_mvtha)
head(data_mvtha,2)
summary(data_mvtha)
str(data_mvtha)

############################## define model ##############################
#diffusion matrix
Mdiffu = matrix(c(0,0,0,0,0,0,
              0,0,0,0,0,0,
              0,0,0,0,0,0,
              0,0,0,0,0,0,
              0,0,0,0,'diffu_dynKBF','diffu_dynBHV_dynKBF',
              0,0,0,0,0,'diffu_dynBHV'), ncol = 6)

#initial cov. matrix fixed to 0 in adequate elements so that these the dynamics processes do not interact or co-vary with the trend processes in the system
Mt0var = matrix(c('T0var_KBF','T0var_BHV_KBF','T0var_cintKBF_KBF','T0var_cintBHV_KBF',0,0,
                  0,'T0var_BHV','T0var_cintKBF_BHV','T0var_cintBHV_BHV',0,0,
                  0,0,'T0var_cintKBF','T0var_cintBHV_cintKBF',0,0,
                  0,0,0,'T0var_cintBHV',0,0,
                  0,0,0,0,'T0var_dynKBF',0,
                  0,0,0,0,'T0var_dynKBF_dynBHV','T0var_dynBHV'), ncol=6)

#The initial state and continuous intercept for each dynamical process is set to zero, to ensure that the processes capture 
#only the fluctuations and not the general trends
Mt0means = matrix(c('T0m_KBF','T0m_BHV','T0m_cintKBF','T0m_cintBHV',0,0),ncol=1)

#do not estimate continuou sintercept paramters as we already have continuous intercept as latent process
MCINT = matrix(c(0,0,0,0,0,0),ncol=1) 

#estimating auto-effect of the dynamics processes
Mdrift = matrix(c('drift_KBF',0,0,0,0,0,
                    0,'drift_BHV',0,0,0,0,
                    1,0,0,0,0,0,
                    0,1,0,0,0,0,
                    0,0,0,0,'drift_dynKBF','drift_dynBHV_dynKBF',
                    0,0,0,0,'drift_dynKBF_dynBHV','drift_dynBHV'), ncol = 6)

#initialize the model with six latent and 2 manifest
model_fit = ctModel(type='stanct', n.manifest=2, n.latent=6, id="id", time="MonthInstudy",
                    manifestNames=c('severity',variable), latentNames=c('KBF','BHV','cintKBF','cintBHV','dynKBF','dynBHV'), 
                    T0MEANS=Mt0means, #initial state of dynamical process set to 0
                    T0VAR=Mt0var,
                    CINT = MCINT,
                    MANIFESTMEANS=matrix(c(0,0),ncol=1),
                    n.TIpred = 4, TIpredNames=c('Dataset1','Dataset3','IsOFH','Is_relocated'),
                    LAMBDA=matrix(c(1,0, 
                                    0,1,
                                    0,0,
                                    0,0,
                                    1,0,
                                    0,1), ncol=6),  
                    DRIFT=Mdrift,
                    tipredDefault=FALSE,
                    DIFFUSION=Mdiffu) 

#we let ind. var in trend. varying initial intercept: T0MEANS
model_fit$pars$indvarying[model_fit$pars$matrix %in% c('T0MEANS')] = TRUE

#group can vary in trend
model_fit$pars$Dataset1_effect[model_fit$pars$matrix %in% c('T0MEANS')] = TRUE
model_fit$pars$Dataset3_effect[model_fit$pars$matrix %in% c('T0MEANS')] = TRUE
model_fit$pars$IsOFH_effect[model_fit$pars$matrix %in% c('T0MEANS')] = TRUE
model_fit$pars$Is_relocated_effect[model_fit$pars$matrix %in% c('T0MEANS')] = TRUE
#check what we did
model_fit$pars[model_fit$pars$matrix %in% c('T0MEANS'),]



############################## fit model ##############################
set.seed(0)
fit_mvtha = ctStanFit(datalong=data_mvtha, ctstanmodel=model_fit, verbose=0, cores=cores_, nopriors=FALSE, #maybe TRUE when optimize ML
                      optimize=TRUE)# add:optimcontrol = list(nsubsets=1) to remove subset warnings
ctsem:::ctSummarise(fit_mvtha, folder=file.path(path_save_,name), cores=cores_, ctStanPlotPost=F, nsamples=1000)
#save model 
save(fit_mvtha,file=file.path(path_save_,'model.rda'))

#to then plot the discrete drift auto and cross effect estimates
df_res = ctStanDiscretePars(fit_mvtha, plot=F, nsamples = 1000, times = seq(from = 0, to = 10, by = 0.1))
dim(df_res)
write.csv(df_res, file=file.path(path_save_,paste0(name,"df_discretedrift.csv")))

#for residual check
df_res = ctKalman(fit_mvtha,subjects=fit_mvtha$setup$idmap[,1],plot=F)#,kalmanvec=c('y','yprior'))
dim(df_res)
write.csv(df_res, file=file.path(path_save_,paste0(name,"df_res_yprior.csv")))




