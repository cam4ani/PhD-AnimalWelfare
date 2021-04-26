import json
import tqdm
import os
import numpy as np
import glob
import pandas as pd
import sys
import shutil
import time
import datetime as dt
import itertools
import time
from IPython.display import HTML as html_print
from collections import defaultdict, Counter
import operator
import scipy
from scipy.stats import entropy
import colorsys
import re
import cv2
import pickle
from operator import itemgetter
import math 

import networkx as nx

from scipy.stats import kurtosis, skew, spearmanr, pearsonr

#time series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf

#PCA
from sklearn.preprocessing import scale
from sklearn import decomposition

#clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.mplot3d import Axes3D

#topics modeling
import pyLDAvis
import pyLDAvis.gensim
import gensim 
from gensim import corpora, models, similarities

#plot
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

#compute simple similarity between two images
from skimage import measure
from skimage.measure import compare_ssim
#other image package
from PIL import Image
import skimage.draw
import imutils

#videos
import imageio
from skimage import color

#change r to 0 from 0.2
sys.path.append('C:\\Users\\camil\\Desktop\\animals_code\\entropy')
from entropy.entropy import sample_entropy
#from entropy.entropy import sample_entropy #https://github.com/raphaelvallat/entropy

#in my understanding the purpose of daily verification is meant to give the possibility that as soon as a new log file arrives one can verify if, wihtout needing to wait for one day. Hence, this daily verification wont induce daily cleaned record, as to be saved it must have the day after too (due to flickering (might need more than one entry, depending on if its in the middle of a flickering situation) and consecutives equal zones (only first entry of next day would be enough))

#Hence, cleaning assumes that the rules are set, while daily verification has the purpose to verify things (and not compute flickering etc) and might typically induce new rules. Hence, for now we have one function to make daily verification, and one to produce cleaned records out of one or more log files. For now if a log file was already cleaned, it will cleaned again, for several reason:
#first it will worth it to not clean again if we have a very long experiemnt, which I am not sure its the case
#then, the log files are not registered at every end day, and hence to clean one log file we need the next one (and not only the first record value, but perhaps plenty, this depends on the longest flickering situations happening at the end of the logfile which might be quite long as we are not necessarily in the night). For this reason, cleaning day by day would be less efficient than in a row (also the way we are handling interzone is way ebtter to do all in ones than each day separately, in this case we would need to change it)
#the number of records lost due to each types of cleaning is print once every run, hence to have these values one must run on the hole logfiles (or change the code)
#--> if we want to compute cleaned record for each log only once, then this should still be implemented. Certainly by given as input the path of the logfiles and a fct to put the df in the correct form, and then removing the log that were already cleaned, opening the rest and clean to put in correct form etc


##########################################################################################################################################
################################################################ others ##################################################################
##########################################################################################################################################
#print with color
def cstr(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s)
def print_color(t):
    display(html_print(' '.join([cstr(ti, color=ci) for ti,ci in t])))
     
        
    
##########################################################################################################################################
############################################################ Preprocessing ###############################################################
##########################################################################################################################################
def preprocessing_broiler_breeder(paths, config, save=True):
    
    '''each experiment should have his own function
    open from a list of csv-path all the csv and aggregated them and put into correct format
    output one df'''
    
    #initialise variables
    path_extracted_data = config.path_extracted_data
    id_run = config.id_run
    date_min = config.date_min
    date_max = config.date_max
    li_henID = config.li_henID
    li_penID = config.li_penID
    dico_zone_matching = config.dico_zone_matching
    dico_matching = config.dico_matching
 
    #create path to save extracted data/info if not existing
    if not os.path.exists(path_extracted_data):
        os.makedirs(path_extracted_data)
    
    li_df = []
    for path in paths:
        df = pd.read_csv(path,sep=';', names=['Timestamp', 'Serialnumber_Tag', 'TagID_Pen_HenID', 'Zone', 'Signal','U1','U2']) 
        #add name of the log to verify issues (e.g. in TagID_Pen_HenID) but mostly to save the cleaned log files only once
        log_name = path.split('\\')[-1].split('.')[0].replace(' ','_')
        df['log_file_name'] = log_name 
        df['log_file_name'] = df['log_file_name'].map(lambda x: int(x.split('_')[1].split('.csv')[0]))
        #TODO: remove if no need of ts_order: add order in timestamp (as there is no miliseconds recorded)
        df['ts_order'] = df.index.copy() 
        li_df.append(df)
    if len(li_df)==0:
        print('No new file to clean'.upper())
        sys.exit()
    df = pd.concat(li_df)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d.%m.%Y %H:%M:%S") #faster with specified format or parse_dates
    df['time'] = df['Timestamp'].map(lambda x: dt.datetime.time(x))
    df['date'] = df['Timestamp'].map(lambda x: dt.datetime.date(x))
    
    #Remove record associated to wrong HenID
    df['TagID_Pen_HenID'] = df['TagID_Pen_HenID'].map(lambda x: x.replace('_',' '))
    x0 = df[df['TagID_Pen_HenID']=='15C3'].shape[0]
    df = df[df['TagID_Pen_HenID']!='15C3']
    print('We remove %d records due to an unkown TagID_Pen_HenID value: 15C3, we are left with %d records'%(x0, df.shape[0]))
    df['HenID'] = df['TagID_Pen_HenID'].map(lambda x: x.split(' ')[-1])
    df['PenID'] = df['TagID_Pen_HenID'].map(lambda x: x.split(' ')[-1][0:-1].strip())
    df['Zone'] = df['Zone'].map(lambda x: x.strip())
    df = df.sort_values(['Timestamp'], ascending=True)    
    #keep only usefull variables and return it
    df = df.filter(['Timestamp', 'HenID', 'Zone','PenID','log_file_name','Signal','ts_order','date','time']).reset_index(drop=True)


    dico_type_li = {'HenID':li_henID,'PenID':li_penID}
    for type_ in ['HenID','PenID']:
        li = dico_type_li[type_]
        df['test_correct_'+type_] = df[type_].map(lambda x: x in li)
        df_ = df[~df['test_correct_'+type_]]
        if df_.shape[0]>0:
            print_color((('There is %d records associated to wrong %s (wrong values: %s)'%(df_.shape[0],type_,
                                                                                      ' /-/ '.join(df_[type_].unique())),'red'),))
        else:
            print_color((('All records are associated to correct '+type_,'green'),))
            l = df[type_].unique()
            if len(l)==len(li):
                print_color((('All '+type_+' has at least one record','green'),))
            else:
                print_color((('BUT not all %s has at least one record (these has no \
                records: %s)'%(type_,' /-/ '.join([i for i in li if i not in l])),'red'),))
    x0 = df.shape[0]
    df = df[(df['test_correct_HenID']) & (df['test_correct_PenID'])]
    print_color((('We have ','black'),(df.shape[0],'green'),(' records (','black'),(x0-df.shape[0],'red'),
                 (' removed due to wrong hen or pen ids)','black')))
    #print('We have %d records (%d removed due to wrong hen or pen ids)'%(df.shape[0],x0-df.shape[0]))   
    
    #######################################################################################################################
    ################# check for zones associated to wrong pen
    if dico_zone_matching!=None:
        print_color((('-----------------------------------------------------------------------------------------------','blue'),))
        print_color((('check zone associated to wrong pen.........','blue'),))
        print_color((('-----------------------------------------------------------------------------------------------','blue'),))

        df_corr = df.groupby(['PenID','Zone']).count().reset_index()
        if save:
            df_corr.to_csv(os.path.join(path_extracted_data, id_run+'_Zone_associated_to_pen_record_numbers.csv'),sep=';')
        #faster than apply : df.apply(lambda x: x['PenID'] in dico_zone_matching[x['Zone']], axis=1)
        df['test_'] = df['PenID']+'/-/'+df['Zone']
        df['test_correct_pen4zone'] = df['test_'].map(lambda x: x.split('/-/')[0] in dico_zone_matching[x.split('/-/')[1]])
        df_corr = df[~df['test_correct_pen4zone']].copy()
        if save:
            df_corr.to_csv(os.path.join(path_extracted_data, id_run+'_Zone_associated_to_wrong_Pen_all_situation.csv'),sep=';')
        x0 = df.shape[0]
        df = df[df['test_correct_pen4zone']]
        print_color((('We have ','black'),(df.shape[0],'green'),(' records (','black'),(x0-df.shape[0],'red'),
                     (' removed due zone associated to wrong pen)','black')))
        df.drop(['test_', 'test_correct_pen4zone'],axis=1,inplace=True)

    ################# now that we have verified zone associated to pen, we can replace the zone by their more general names
    df['Zone'] = df['Zone'].map(lambda x: dico_matching[x])       
       
    #choose the date you want if any
    if date_min!=None:
        print(date_min)
        print('lets look at the record only between date %s and %s'%(str(date_min),str(date_max)))
        df = df[(df['Timestamp']>=date_min) & (df['Timestamp']<=date_max)]

    if save:
        df.to_csv(os.path.join(path_extracted_data, id_run+'_PreprocessRecords.csv'), sep=';')

    return(df)  


 
##########################################################################################################################################
########################################################## 1 sec time series #############################################################
##########################################################################################################################################
    
def time_series_henColumn_tsRow(df, config, col_ts='Zone' , name_='', hen_time_series=False):
    
    '''one time series with each column being one hen. because then opening one file we have all. also, no need to go column by column to change day'''
    
    #start recording the time it last
    START_TIME = time.clock()        

    #initialize parameter
    path_extracted_data = config.path_extracted_data
    id_run = config.id_run
    nbr_sec = 1 #should stay one
    
    #create a director if not existing
    if not os.path.exists(path_extracted_data):
        os.makedirs(path_extracted_data)
        
    #verify columns name if not done before: TODO
    
    print('in this time series there is %d hens'%len(df['HenID'].unique()))
    
    #remove equal timestamps records for same hens
    df.sort_values(['Timestamp','ts_order'], ascending=True, inplace=True)
    df.drop_duplicates(subset=['Timestamp','HenID'], keep='last', inplace=True)
    
    #add hen_ in front of the hen ids, as it will help for later (slecting each column associated to a hen)
    df['HenID'] = df['HenID'].map(lambda x: 'hen_'+x)

    #sort by timestamp and do pivot
    df = df.sort_values(['Timestamp'], ascending=True)
    x0 = df.shape[0]

    #No need now to have no duplicate timestamp per hen:
    #df = df.groupby(['Timestamp','HenID'])[col_ts].agg(lambda x: list(x)[0]).reset_index()  
    df_hens = df.pivot(index='Timestamp', columns='HenID', values=col_ts)

    #fill "None" values with the last non-empty value (by propagating last valHenID observation forward to next valHenID)
    #In order to fill in between timestamp, ie. timestamp that another hen had, then the other should also have their latest zone 
    #entered instead of nan Note that the first ones will stay None
    df_hens = df_hens.fillna(method='ffill')

    #Warning: not all hens have same initial/enddate!
    #add missing dates 
    mi = min(df['Timestamp'].tolist())
    ma = max(df['Timestamp'].tolist())
    print('The initial starting date in over all is: %s, and the ending date will be: %s'%(str(mi), str(ma)))
    print('But note that birds may have different ending and starting date which should be taken into account when computing variables')
    
    #add dates until minuit of the last day
    ma = dt.datetime(ma.year,ma.month,ma.day,23,59,59)
    print('and after ending the last day at midnight : %s, and the ending date will be: %s'%(str(mi), str(ma)))
    Daterange = pd.date_range(start=mi, end=ma, freq='S') 

    #take only the needed values (nbr_sec)
    Daterange = [Daterange[i] for i in range(len(Daterange)) if i%nbr_sec==0]
    #Daterange[0:10]
    
    #add missing seconds (i.e. all seconds that never had a record) and fillnan with last non-nan values by propagating last 
    #valHenID observation (even if its an observation that will be removed) forward to next valHenID
    df_hens = df_hens.reindex(Daterange, method='ffill').reset_index()
    #df_hens.tail(20)
    
    df_hens['date'] = df_hens['Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day))
    print('-------------- Lets save')
    print(df_hens.shape)
    df_hens.to_csv(os.path.join(path_extracted_data,id_run+'_TimeSeries'+str(name_)+'.csv'), sep=';', index=False)

    #one time serie per hen
    if hen_time_series:
        print('-------------- Lets compute individuals seconds time series')
        #create a director if not existing
        path_ts = os.path.join(config.path_extracted_data, 'HenInitial_1secTs')
        if not os.path.exists(path_ts):
            os.makedirs(path_ts)
        #remove the date keep only the timestamp
        df_hens['Timestamp_value'] = df_hens['Timestamp'].map(lambda x: dt.timedelta(hours=x.hour, 
                                                                                     minutes=x.minute, 
                                                                                     seconds=x.second))        
        li_hen = [h for h in df_hens.columns if h.startswith('hen_')]
        for h in tqdm.tqdm(li_hen):
            #select the column associated to the hen
            df_per_hen = df_hens[[h,'Timestamp_value','date']].copy()
            #pivot, to put the date in column intead of having one row for each timestamp_value per date
            df_per_hen = df_per_hen.pivot(index='Timestamp_value', columns='date', values=h)
            df_per_hen.reset_index(drop=False, inplace=True)
            df_per_hen.to_csv(os.path.join(path_ts, id_run+'_TimeSeries_initial_'+str(name_)+'_'+h+'.csv'), sep=';')
    
    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))  
    
    return(df_hens)
    
    
    
    
##########################################################################################################################################
################################################################# bining #################################################################
##########################################################################################################################################
    
    
def bining_broilers(df_ts, config, nbr_sec_mean, mi=None, ma=None, bining_1sec_ts=False):
    
    ''' 
    *input: nbr_sec_mean: period, df_ts: time serie dataframe, typically created by the function "time_series_henColumn_tsRow()"
    *output: a csv where timestamp ts results in the bining the all record from ts-period to ts]
    *main idea: create time series for each hen by taking the most frequent zone for each "nbr_sec_mean" seconds period
    *programming main idea: First we create a list of timestamp including only the one we want (i.e. one per nbr_sec_mean seconds). Then we match the old timestamp with the smallest of the list taht is beger of equal to the actual timestamp
    '''
    
    #start recording the time it last
    START_TIME = time.clock()
      
    #initialize parameters
    id_run = config.id_run
    path_extracted_data = config.path_extracted_data
    nbr_sec = 1 #should stay one for now
    
    #create a directory if not existing
    path_ = os.path.join(path_extracted_data, 'HensBiningTimeSeries')
    if not os.path.exists(os.path.join(path_)):
        os.makedirs(os.path.join(path_))
        
    #######################################################################################################################
    ##### create a list of dates that we want starting from our initial and end dates with the wanted binning period ######
    if mi==None:
        mi = min(df_ts['Timestamp'].tolist())
    if ma==None:
        ma = max(df_ts['Timestamp'].tolist())
    #keeping dataframe that is linked to these dates
    df_ts = df_ts[(df_ts['Timestamp']>=mi) & (df_ts['Timestamp']<=ma)]
    
    #on arondi a la minute du bas/haut pour ne pas rater des records 
    Daterange = pd.date_range(start = mi-dt.timedelta(seconds=mi.second), 
                              end = ma+dt.timedelta(seconds=60-ma.second), 
                              freq = 'S')    
    print('The starting date of the datetime list is: %s, and the ending date is: %s'%(str(Daterange[0]), str(Daterange[-1])))
    #take only the wanted timestamps (nbr_sec_mean)
    Daterange = [Daterange[i] for i in range(len(Daterange)) if i%nbr_sec_mean==0]
    print('The starting date of the selected datetime list is: %s, and the ending date is: %s'%(str(Daterange[0]), str(Daterange[-1])))

    #######################################################################################################################
    #add new timestamp to the initial file
    df_date = pd.DataFrame({'New_Timestamp':Daterange})
    df_date['New_Timestamp'] = df_date['New_Timestamp'].map(lambda x: pd.to_datetime(x))
    df_ts = pd.merge_asof(df_ts, df_date, left_on=['Timestamp'], right_on=['New_Timestamp'], direction='forward')
    
    #compute nbr_sec computation here (list of difference between each timestamp, and must always be the same)
    li_ts = df_ts['Timestamp'].tolist()
    li_diff_ts = list(set(list(map(operator.sub, li_ts[1:], li_ts[0:-1]))))
    if len(li_diff_ts)!=1:
        print('ERROR: your timestamp columns have different one to one difference: ', li_diff_ts)
        sys.exit()
    nbr_sec = li_diff_ts[0].total_seconds()
    if nbr_sec!=1:
        print(nbr_sec)
        print('ERROR: your time series has %d seconds between two timestamps, it should be one seconds!'%nbr_sec)     
        sys.exit()
    df_ts['nbr_sec'] = nbr_sec # or count() instead of sum and rename
    #df_ts['New_Timestamp_old'] = df_ts['Timestamp'].map(lambda x: min([d for d in Daterange if d >= x], default=np.nan))
    
    #aggregate (by using groupby: for each hen take its time serie and find the most frequent zone per new_timestamp)
    for h in [x for x in df_ts.columns if x.startswith('hen_')]:
        df_ = df_ts[[h,'New_Timestamp','nbr_sec','Timestamp']].copy()
        #df_verification = df_ts.groupby(['New_Timestamp']).agg({'Timestamp':['max', 'min']}).reset_index()
        #df_verification.to_csv(os.path.join(path_ ,id_run+'_ts_MostFrequentZone_period_VERIFICATION'+str(nbr_sec_mean)+'_'+str(mi).split(' ')[0]+'_'+str(ma).split(' ')[0]+'_'+h+'.csv'), sep=';', index=False)

        df__ = df_.groupby(['New_Timestamp',h]).agg(
                   nbr_sec=pd.NamedAgg(column='nbr_sec', aggfunc=lambda x: sum(x)),
                   first_timestamp=pd.NamedAgg(column='Timestamp', aggfunc=lambda x: min(x))).reset_index()
        df_final = df__.groupby(['New_Timestamp'])[[h,'nbr_sec','first_timestamp']].agg(lambda x: tuple(x)).reset_index()
        df_final['nbr_zones'] = df_final[h].map(lambda x: len(x))
        #tuples are ordered sequence of elements
        df_final['dico_zone_dur_timestamp'] = df_final.apply(lambda x: {x[h][k]:(x['nbr_sec'][k],
                                                                           x['first_timestamp'][k]) for k in range(len(x[h]))}, 
                                                       axis=1)
        df_final['dico_zone_dur'] = df_final.apply(lambda x: {x[h][k]:x['nbr_sec'][k] for k in range(len(x[h]))}, 
                                                       axis=1)
        df_final['dico_zone_timestamp'] = df_final.apply(lambda x: {x[h][k]:x['first_timestamp'][k] for k in range(len(x[h]))}, 
                                                       axis=1)
        df_final['most_frequent_zones'] = df_final['dico_zone_dur'].map(lambda x: [k for k, v in x.items() if v==max(x.values())])
        df_final['nbr_mf_zones'] = df_final['most_frequent_zones'].map(lambda x: len(x))
        df_final['most_frequent_zone'] = df_final.apply(lambda x: [k for k, v in x['dico_zone_timestamp'].items() if v==min([x['dico_zone_timestamp'][i] for i in x['most_frequent_zones']])], axis=1)

        #small verification
        df_final['verificiation_nbr'] = df_final['most_frequent_zone'].map(lambda x: len(x))
        if df_final[df_final['verificiation_nbr']>1].shape[0]>0:
            print('ERROR')
            sys.exit()
        df_final.drop(['verificiation_nbr'],inplace=True,axis=1)

        df_final['most_frequent_zone'] = df_final['most_frequent_zone'].map(lambda x: x[0])

        #compute small stats
        df_final['lost_duration_per_zone'] = df_final.apply(lambda x:  {k:v for k,v in x['dico_zone_dur'].items() if k!=x['most_frequent_zone']}, axis=1)
        df_final['lost_duration'] = df_final['lost_duration_per_zone'].map(lambda x: sum(x.values()))
        df_final['perc_lost_duration'] = df_final['lost_duration'].map(lambda x: x/nbr_sec_mean*100)
        df_final['day'] = df_final['New_Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day))
        
        df_final.to_csv(os.path.join(path_ ,id_run+'_ts_MostFrequentZone_period'+str(nbr_sec_mean)+'__'+str(mi).split(' ')[0]+\
                                     '_'+str(ma).split(' ')[0]+'_'+h+'.csv'), sep=';', index=False)      
        
        if bining_1sec_ts:
            #create a director if not existing
            path_ts = os.path.join(config.path_extracted_data, 'HenBining_1secTs')
            if not os.path.exists(path_ts):
                os.makedirs(path_ts)
            df_1ts = df_final[['most_frequent_zone','New_Timestamp']].copy()
            #add dates until minuit of the last day
            ma = dt.datetime(ma.year,ma.month,ma.day,23,59,59)
            Daterange = pd.date_range(start=mi, end=ma, freq='S')
            #add missing seconds (i.e. all seconds that never had a record) and fillnan with last non-nan values by propagating last 
            #valHenID observation (even if its an observation that will be removed) forward to next valHenID
            df_1ts.set_index('New_Timestamp', inplace=True)
            #backward fill, as the binging time series of time t is defined on ]t-period,t]
            df_1ts = df_1ts.reindex(Daterange, method='bfill').reset_index()
            df_1ts.to_csv(os.path.join(path_ts, id_run+'_TimeSeries_bining_'+str(nbr_sec_mean)+'__'+str(mi).split(' ')[0]+\
                                     '_'+str(ma).split(' ')[0]+'_'+h+'.csv'), sep=';')

    
    #running time info and return final cleaned df
    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))
    
    return

    
    
    
    
    
    
    
    
    
    
    
    
    