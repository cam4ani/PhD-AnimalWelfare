library(dplyr) #%>%


################################################
################ download data #################
################################################
path_ = 'G:/VPHI/Welfare/2- Research Projects/OFHE2.OriginsE2/DataOutput/TrackingSystem/ALLDATA_'
path_visual = file.path(path_, 'OFH_performance')
dir.create(path_visual)
df = read.csv(file.path(path_,'df_eggdata.csv'), header = TRUE, sep = ",")
df$PenID = factor(df$PenID)
df$DIB_f = factor(df$DIB)
df$WIB_f = factor(df$WIB)
df$DIB = as.integer(df$DIB)
df$Treatment_allpens = factor(df$Treatment_allpens) #unsacaled: 0-1
#setting reference group
df = df %>% mutate(Treatment_allpens = relevel(Treatment_allpens, ref = "TRAN"))
contrasts(df$Treatment_allpens)
print(dim(df))
summary(df)
head(df,3)


#choosing first two month in the laying barn
df_egg = df[(df$DIB<=60)&(df$DIB>=1),]


################################################
################# sigmoid curve ################
################################################
li = list()
tiff(file.path(path_visual, 'egg_growth_crv.tiff'), width=400, height=400)
df_OFH = df_egg[(df_egg$Treatment_allpens=='OFH'),]
df_TRAN = df_egg[(df_egg$Treatment_allpens=='TRAN'),]
plot(df_OFH$DIB, df_OFH$X.eggPerTier, col=rgb(red = 0, green = 0, blue = 1, alpha = 0.1))
points(df_TRAN$DIB, df_TRAN$X.eggPerTier, col=rgb(red = 1, green = 0, blue = 0, alpha = 0.1))
#iterate for each pen: one curve / pen
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
    #"The logistic growth curve has some upper bound on it (i.e. some point which the model can not pass). The model estimates this
    #to be 1858.0956. So, if I were to make predictions for very large x, you would see that the curve will get very close to 25.657 but will never touch it or pass it. T
    #plot(df_penid$DIB, df_penid$X.eggPerTier)
    lines(df_penid$DIB, predict(fit_penid), col=color, lwd=1.2)  
    df_res = as.data.frame(summary(fit_penid)$coefficients)
    df_res$PenID = penid
    li[[i]] = df_res
    i = i + 1
}

dev.off()
df_res = rbind(li[[1]],li[[2]],li[[3]],li[[4]],li[[5]],li[[6]],li[[7]],li[[8]],li[[9]],li[[10]])
write.csv(df_res, file.path(path_visual,'egg_growthcurve.csv')) 
df_res






































