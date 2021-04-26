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
from collections import defaultdict, Counter
import operator
import re
import cv2
import pickle
from operator import itemgetter
import math 
from IPython.display import HTML as html_print


############################################################################################################################################################################################################### step 1 #################################################################
############################################################################################################################################

#match the tracking system to the correct henid and small verification. 
#Important: FROM THIS POINT; ONLY WORK WITH HENID; NOT TAGID

def preprocessing_experiment2(paths, path_FocalBird, config):
    
    '''each experiment should have his own function
    This function opens (from a list of csv-path) all the csv, aggregate them into correct format'''

    ####################################################################################
    ############################### Initialise variables ###############################
    ####################################################################################
    path_extracted_data = config.path_extracted_data
    id_run = config.id_run
    date_min = config.date_min
    date_max = config.date_max

    #create path to save extracted data/info if not existing
    if not os.path.exists(path_extracted_data):
        os.makedirs(path_extracted_data)

        
    ####################################################################################
    ####### Download all logs one by one adding the logfilename and the ts_order #######
    ####################################################################################
    li_df = []
    for path_ in paths:
        log_name = path_.split('\\')[-1].replace(' ','').replace('.','').split('.csv')[0]
        df = pd.read_csv(path_, sep=';', names=['Timestamp', 'SerialNum', 'TagID', 'Zone', 'Marker','U1','U2']) 
        df['log_file_name'] = log_name 
        df['ts_order'] = df.index.copy() 
        df['system'] = path_.split('TagUpdates')[1].split('\\')[0]
        v = df.shape[0]
        if v<80000:
            print_color((('log: %s has '%log_name,'black'),(v,'red'),(' rows','black')))
        else:
            print_color((('log: %s has '%log_name,'black'),(v,'green'),(' rows','black')))
        li_df.append(df)
    df = pd.concat(li_df)
    #remove the records with no timestamp (appearing at least in the first log of pen 3-5, named *_NA)
    df = df[~df['Timestamp'].isin(['0','1','2','3','4','5','6','7','8','9','10'])]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d.%m.%Y %H:%M:%S") #faster with specified format than parse_dates
    df['time'] = df['Timestamp'].map(lambda x: dt.datetime.time(x))
    df['date'] = df['Timestamp'].map(lambda x: dt.datetime.date(x))
    df['TagID'] = df['TagID'].astype(str)

    
    ####################################################################################
    ############### Download info on henID associtation to (TagID,date) ################
    ####################################################################################
    df_FB = pd.read_excel(path_FocalBird, parse_dates=['StartDate','EndDate'])
    #fill end date to today+1 for the birds which we dont know when is there end date (+1: so that today is taken into account)
    print(date_max+dt.timedelta(days=1))
    df_FB['EndDate'].fillna(date_max+dt.timedelta(days=1), inplace=True)
    df_FB['TagID'] = df_FB['TagID'].astype(str)
    #exclude rows were tags were not functionning correctly for some reason 
    df_FB = df_FB[df_FB['ShouldBeExcluded']!='yes']


    ####################################################################################
    ####################### Add a unique HenID to tracking data ########################
    ####################################################################################   
    #transform into one row per date per tagID
    li_dico = []
    for i in range(df_FB.shape[0]):
        x = df_FB.iloc[i]
        li_dates = pd.date_range(start=x['StartDate']+dt.timedelta(days=1), 
                                 end=x['EndDate']-dt.timedelta(days=1), freq='D')
        for d in li_dates:
            dico_ = dict(x)
            dico_['date'] = d
            li_dico.append(dico_)
    df_FB_daily = pd.DataFrame(li_dico)
    #ensure date has no h/m/s
    df_FB_daily['date'] = df_FB_daily['date'].map(lambda x: dt.datetime.date(x))

    #merge tracking data with hens info
    df = pd.merge(df, df_FB_daily[['HenID','PenID','date','TagID']], on=['date','TagID'], how='inner') 
    #small verification:
    #df[(df['HenID'].isnull())&(df['TagID']=='15')]['date'].unique()
    #note that : how=inner in order to oly have records that are correctly associated to a chicken
    #how!= left as we need to remove some records if the system was resetting etc, so we dont want to keep the tracking data of 
    #tags that were not working correctly on that day
    df_FB_daily.to_csv(os.path.join(path_extracted_data, id_run+'_df_FB_daily.csv'),sep=';')

    
    ####################################################################################
    ##################### Verify if each hen is in the correct pen #####################
    ####################################################################################      
    df_ = df.groupby(['HenID'])['system','PenID','TagID'].agg(lambda x: set(x)).reset_index()
    df_['nbr_system'] = df_['system'].map(lambda x: len(x))
    df_['li_system'] = df_['system'].map(lambda x: list(range(int(list(x)[0].split('_')[0].strip()), 
                                                              int(list(x)[0].split('_')[1].strip())+1)))
    df_['correct_pen'] = df_.apply(lambda x: int(list(x['PenID'])[0]) in x['li_system'], axis=1)
    display(df_.head(3))
    if df_[df_['nbr_system']>1].shape[0]!=0:
        print('ERROR: some hens belong to two system:')
        display(df_[df_['nbr_system']>1])
        return df
        #sys.exit()
    if df_[~df_['correct_pen']].shape[0]!=0:
        print('ERROR: some hens belong to the INCORRECT systems:')
        display(df_[~df_['correct_pen']])
        return df
        #sys.exit()
    
    ####################################################################################
    ########################### Checks types, select column ############################
    ####################################################################################   
    df['PenID'] = df['PenID'].map(int).map(str)
    df['Zone'] = df['Zone'].map(lambda x: x.strip())
    df = df.sort_values(['Timestamp'], ascending=True)
    df = df.filter(['Timestamp', 'HenID', 'Zone','PenID','log_file_name','ts_order','TagID','date','time']).reset_index(drop=True)
    
    ####################################################################################
    ################# remove dates outside of datemin_date_max & save ##################
    #################################################################################### 
    if date_min!=None:
        print(date_min)
        print('lets look at the record only between date %s and %s'%(str(date_min),str(date_max)))
        df = df[(df['Timestamp']>=date_min) & (df['Timestamp']<=date_max)]

    df.to_csv(os.path.join(path_extracted_data, id_run+'_PreprocessRecords.csv'),sep=';')
    
    return(df)



##################################################################################################################################################################################################### additionalstep if wanted #########################################################
############################################################################################################################################

#the cleaning that is general to all UWB systems: records with exact same timestamp, would have a microseconds added, so that they both will be taken into account

def general_cleaning(df, config, save=True):
    
    '''
    *Input: dataframe with the following columns: ['Timestamp','Zone','HenID','PenID','log_file_name','ts_order'], typically
    created by a "preprocessing_*()" function. The ts_order columns is the index of each different log file
    *assumption: the log_file_name order correspond to the correct date order
    *Why are we doing this: TO remove the equaltimestamp with different zone for the same hen, as this is due to how the system is recording the data. We do it that way, first in order to have a simpler code and easier to understand as it will remove all situation with equal timestamp different zones and the ts_order information. Second, to avoid mistakes (e.g. induce by two equal timestamp with different zone  that appear in different logfile and hence ts_order is different)
    *Main programming idea: looping over hen and add the microseconds depending on the number of records associated to this exact timestamp and to its order of appearence in time. In other words, I add microseconds value in the timestamp (the following way: 1000000/(nbr equal timestamp records)*record_order, where record_order starts at 0 and knowing that 1sec=1'000'000microseconds and that microseconds is the value that appear after seconds in datetime()
    *Exemple 1: 
    record x hen w: 2019-11-24 23:42:09.000000 --> 2019-11-24 23:42:09.000000
    record y hen w: 2019-11-24 23:42:09.000000 --> 2019-11-24 23:42:09.500000
    *Exemple 2: 
    record x hen w: 2019-11-24 23:42:09.000000 --> 2019-11-24 23:42:09.000000
    record y hen w: 2019-11-24 23:42:09.000000 --> 2019-11-24 23:42:09.333333
    record z hen w: 2019-11-24 23:42:09.000000 --> 2019-11-24 23:42:09.666666
    *Output: the exact same csv, with few new columns, Timestamp columns has the microseconds and Initial_timestamp is the initial one (i.e. without microseconds)'''
    
    #start recording the time it last
    START_TIME = time.clock()
    
    #######################################################################################################################    
    ################# verify the columns name and types
    li_colnames = ['Timestamp','Zone','HenID','PenID','log_file_name','ts_order','date']
    if not all(i in df.columns for i in li_colnames):
        print('Check your column names, they should include: ',' '.join(li_colnames))
        sys.exit()
    #types
    df = df.astype({'HenID': 'str', 'PenID': 'str', 'Zone': 'str', 'log_file_name': 'str', 'ts_order': 'str'})
    if not df.dtypes['Timestamp']=='datetime64[ns]':
        print('ERROR: Timestamp column should be of type datetime64[ns]')
        sys.exit()
        
    #add some info making thing more clear and general
    df['HenID'] = df['HenID'].map(lambda x: 'hen_'+x)
    
    ################# import parameters from configuration file
    id_run = config.id_run
    path_extracted_data = config.path_extracted_data
    #create path if it does not exist
    if not os.path.exists(path_extracted_data):
        os.makedirs(path_extracted_data)
    dico_zone_matching = config.dico_zone_matching
    dico_matching = config.dico_matching

    #######################################################################################################################
    ################# remove zone associated to wrong pen
    if dico_zone_matching!=None:
        print_color((('-----------------------------------------------------------------------------------------------','blue'),))
        print_color((('remove zone associated to wrong pen.........','blue'),))
        print_color((('-----------------------------------------------------------------------------------------------','blue'),))

        df_corr = df.groupby(['PenID','Zone']).count().reset_index()
        if save:
            df_corr.to_csv(os.path.join(path_extracted_data, id_run+'_Zone_associated_to_pen_record_numbers.csv'),sep=';')
        #faster than apply : df.apply(lambda x: x['PenID'] in dico_zone_matching[x['Zone']], axis=1)
        df['test_'] = df['PenID']+'/-/'+df['Zone']
        df['test_correct_pen4zone'] = df['test_'].map(lambda x: x.split('/-/')[0] in dico_zone_matching[x.split('/-/')[1]])
        df_corr = df[~df['test_correct_pen4zone']]
        if save:
            df_corr.to_csv(os.path.join(path_extracted_data, id_run+'_Zone_associated_to_wrong_Pen_all_situation.csv'),sep=';')
        x0 = df.shape[0]
        df = df[df['test_correct_pen4zone']]
        print_color((('We have ','black'),(df.shape[0],'green'),(' records (','black'),(x0-df.shape[0],'red'),
                     (' removed due zone associated to wrong pen)','black')))

    ################# now that we have verified zone associated to pen, we can replace the zone by their more general names
    df['Zone'] = df['Zone'].map(lambda x: dico_matching[x])

    #######################################################################################################################
    ################# remove ts_order and add miliseconds
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    print_color((('remove ts_order and add miliseconds.........','blue'),))
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    df['ts_order_logname'] = df['log_file_name'].map(str)+'_'+df['ts_order'].map(str)
    #removing whitespace in ts_order_logname, which is important for ms calculation as it wont work if log_file_name has some space in it
    df['ts_order_logname'] = df['ts_order_logname'].map(lambda x: x.replace(' ','_')) 
    #keep track of the initial timestamp for verification
    df['Timestamp_initial'] = df['Timestamp'].copy()
    #loop over each hen and add the correct milliseconds
    li_df = []
    for i, df_hen in tqdm.tqdm(df.groupby(['HenID'])):
        #get all ts_order_logname info appearing on the exact same timestamp into a dictionary and match it to each timestamp value
        #to have a new column with the list of all the "ts_order_logname" occruign at this timestamp (more efficient than apply)
        df_ = df_hen.groupby(['Timestamp', 'HenID'])['ts_order_logname'].agg(lambda x: ' '.join(x)).reset_index()
        dico_ts_tuple_tsorder_zone = dict(zip(df_['Timestamp'], df_['ts_order_logname']))
        df_hen['ts_order_list'] = df_hen['Timestamp'].map(lambda x: dico_ts_tuple_tsorder_zone[x])
        #add the ts_order_logname associated to that record so that it appears 2 times
        df_hen['ts_order_info'] = df_hen['ts_order_list'].copy().map(str)+' '+df_hen['ts_order_logname'].copy() 

        #take the index from the sorted (smaller to bigger) list of the ts_order_logname value that is associated to this record
        #(i.e. appear two times (i.e. max))
        #1sec=1'000'000microseconds, microseconds is the value that appear after seconds in datetime()
        df_hen['ms'] = df_hen['ts_order_info'].map(lambda x: sorted(Counter(x.split(' ')).keys()).index(max(Counter(x.split(' ')).items(),
                                                                    key=operator.itemgetter(1))[0])*(1000000/float(len(x.split(' '))-1)))
        df_hen['Timestamp'] = df_hen.apply(lambda x: x['Timestamp']+dt.timedelta(microseconds=int(x['ms'])),axis=1)
        li_df.append(df_hen)
    #put again in one dataframe
    df = pd.concat(li_df)
    
    ################# save
    df = df.sort_values(['Timestamp'], ascending=True)
    df = df.filter(['Timestamp','HenID','Zone','PenID','log_file_name','date','ts_order_logname','ts_order_list','ms',
                     'Timestamp_initial'])
    if save:
        df.to_csv(os.path.join(path_extracted_data, id_run+'_records_GeneralCleaning.csv'), sep=';', index=False) 
              
    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))  
    
    return(df)



############################################################################################################################################
################################################################## Bining ##################################################################
############################################################################################################################################

#split into two functions, one that create one second time serie and one that use it to aggregate for the bining. Only after this step, when we create variables, the specific days per hen should be removed. Not 

def time_series_henColumn_tsRow(df, config, col_ts='Zone' , name_=''):
    
    '''one time series with each column being one hen. because then opening one file we have all. also, no need to go column by column to change day'''
    
    #start recording the time it last
    START_TIME = time.clock()        

    #initialize parameter
    path_extracted_data = config.path_extracted_data
    id_run = config.id_run
    
    #create a director if not existing
    if not os.path.exists(path_extracted_data):
        os.makedirs(path_extracted_data)    
    print('in this time series there is %d hens'%len(df['HenID'].unique()))
    
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
    print('But note that birds may have different ending and starting date, we remove these dates when computing the daily variables')
    
    #add dates until minuit of the last day
    ma = dt.datetime(ma.year,ma.month,ma.day,23,59,59)
    print('and after ending the last day at midnight : %s, and the ending date will be: %s'%(str(mi), str(ma)))
    Daterange = pd.date_range(start=mi, end=ma, freq='S') 
    
    #add missing seconds (i.e. all seconds that never had a record) and fillnan with last non-nan values by propagating last 
    #valHenID observation (even if its an observation that will be removed) forward to next valHenID
    df_hens = df_hens.reindex(Daterange, method='ffill').reset_index()
    #df_hens.tail(20)
    df_hens.rename(columns={'index':'Timestamp'}, inplace=True)
    df_hens['date'] = df_hens['Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day))
    
    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))  
    
    return(df_hens)



def bining(df, config, nbr_sec_mean, mi=None, ma=None, save=True):
    
    ''' 
    *input: nbr_sec_mean: period, df: data after the general cleaning"
    *output: one csv per hen where timestamp ts results in the bining of records from ts-period to ts]
    *main idea: create time series for each hen by taking the most frequent zone for each "nbr_sec_mean" seconds period
    *programming main idea: First we create a list of timestamp including only the one we want (i.e. one per nbr_sec_mean seconds). Then we match the old timestamp with the smallest of the list that is beger of equal to the actual timestamp
    '''
    
    #start recording the time it last
    START_TIME = time.clock()
    
    print('create time series')
    df_ts = time_series_henColumn_tsRow(df, config, col_ts='Zone')
    print('finish creating time series')
      
    #initialize parameters
    id_run = config.id_run
    path_extracted_data = config.path_extracted_data
    
    #create a directory if not existing
    path_ = os.path.join(path_extracted_data, 'HensTimeSeries')
    if not os.path.exists(os.path.join(path_)):
        os.makedirs(os.path.join(path_))
        
    #######################################################################################################################
    ##### create a list of dates that we want starting from our initial and end dates with the wanted binning period ######
    mi = min(df_ts['Timestamp'].tolist())
    ma = max(df_ts['Timestamp'].tolist())
    #keeping dataframe that is linked to these dates
    #df_ts = df_ts[(df_ts['Timestamp']>=mi) & (df_ts['Timestamp']<=ma)]
    
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
    #df_ts['New_Timestamp_old'] = df_ts['Timestamp'].map(lambda x: min([d for d in Daterange if d >= x], default=np.nan))
    
    #aggregate (by using groupby: for each hen take its time serie and find the most frequent zone per new_timestamp)
    for h in tqdm.tqdm([x for x in df_ts.columns if x.startswith('hen_')]):
        df_ = df_ts[[h,'New_Timestamp']].copy()
        df_['nbr_sec'] = 1
        #df_verification = df_ts.groupby(['New_Timestamp']).agg({'Timestamp':['max', 'min']}).reset_index()
        #df_verification.to_csv(os.path.join(path_ ,id_run+'_ts_MostFrequentZone_period_VERIFICATION'+str(nbr_sec_mean)+'_'+str(mi).split(' ')[0]+'_'+str(ma).split(' ')[0]+'_'+h+'.csv'), sep=';', index=False)
        df__ = df_.groupby(['New_Timestamp',h])['nbr_sec'].sum().reset_index() #sum to count as we have seconds
        df_final = df__.groupby(['New_Timestamp'])[h,'nbr_sec'].agg(lambda x: tuple(x)).reset_index()
        df_final['most_frequent_zone'] = df_final.apply(lambda x: x[h][x['nbr_sec'].index(max(x['nbr_sec']))], axis=1)
        df_final['nbr_duration_per_zone'] = df_final.apply(lambda x: str({x[h][k]:x['nbr_sec'][k] for k in range(len(x[h]))}), axis=1)
        df_final['nbr_lost_duration_per_zone'] = df_final['nbr_duration_per_zone'].map(lambda x: str({z:v for z,v in eval(x).items() if \
                                                                           v!=max(eval(x).values())}))
        df_final['nbr_lost_duration'] = df_final['nbr_lost_duration_per_zone'].map(lambda x: sum(eval(x).values()))
        df_final['perc_lost_duration'] = df_final['nbr_lost_duration'].map(lambda x: x/nbr_sec_mean*100)
        df_final['day'] = df_final['New_Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day))
        if save:
            df_final.to_csv(os.path.join(path_, id_run+'_ts_MostFrequentZone_period'+str(nbr_sec_mean)+'_'+str(mi).split(' ')[0]+\
                                         '_'+str(ma).split(' ')[0]+'_'+h+'.csv'), sep=';', index=False)
    
    #running time info and return final cleaned df
    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))
    
    return




############################################################################################################################################
######################################################### Variables TODO IF NEEDED #########################################################
############################################################################################################################################

def HenVariable(df, config, path_FocalBird, ts_name, name_='', timestamp_name='Timestamp'):
    
    #to add if needed
        
    ################################################################################################################
    ############################# add basics hens info, remove unwanted dates and save #############################
    ################################################################################################################
            
    #add basics hens info
    #download info on henID associtation to (TagID,date) 
    df_FB = pd.read_excel(path_FocalBird, parse_dates=['StartDate','EndDate'])
    df_FB['HenID'] = df_FB['HenID'].map(lambda x: 'hen_'+str(x))
    df_FB = df_FB[df_FB['ShouldBeExcluded']!='yes']
    df_FB['EndDate'].fillna(date_max+dt.timedelta(days=1), inplace=True)
    
    #Note: the HenID was already match according to the correct dates. 
    #Assumption:Each henID is linked to a unique PenID!
    df_daily = pd.merge(df_daily, df_FB[['HenID','PenID','CLASS','29-09 weight']], on=['HenID'], how='left')

    #remove dates with health care
    print('-------------- Lets remove unwanted dates that impacted ALL PENS')
    if len(li_date2remove)!=0:
        df_daily['date_toberemoved'] = df_daily['level'].map(lambda x: x in li_date2remove)
        x0 = df_daily.shape[0]
        df_daily = df_daily[~df_daily['date_toberemoved']]
        print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                    df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))    

    #remove dates linked to specific pens
    print('-------------- Lets remove unwanted dates that impacted FEW PENS')
    if len(dico_date2remove_pens)!=0:
        df_daily['date_2remove_penper'] = df_daily.apply(lambda x: int(x['PenID']) in dico_date2remove_pens[x['level']], axis=1)
        x0 = df_daily.shape[0]
        df_daily = df_daily[~df_daily['date_2remove_penper']]
        print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                    df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))   
        
    #remove dates linked to specific hens
    print('-------------- Lets remove dates that impacted FEW HENS')
    #create a dictionary with henID as keys and a list of tracking-active days
    dico_hen_activedate = defaultdict(list)
    for i in range(df_FB.shape[0]):
        x = df_FB.iloc[i]
        li_dates = pd.date_range(start=x['StartDate']+dt.timedelta(days=1), 
                                 end=x['EndDate']-dt.timedelta(days=1), freq='D')
        dico_hen_activedate[x['HenID']].extend([dt.datetime.date(d) for d in li_dates])
    df_daily['level'] = df_daily['level'].map(lambda x: dt.datetime.date(x))
    df_daily['date_2remove_penhen'] = df_daily.apply(lambda x: x['level'] not in dico_hen_activedate[x['HenID']], axis=1)
    x0 = df_daily.shape[0]
    df_daily = df_daily[~df_daily['date_2remove_penhen']]
    print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))      

    #save
    df_daily.drop(['verification_daily_total_nbr_hour','zone_list'],inplace=True,axis=1) #verification_daily_total_duration
    df_daily.to_csv(os.path.join(path_extracted_data, id_run+'_'+ts_name+'_'+name_+'_variables.csv'), sep=';', index=False)

    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))

    
    return(df_daily)


##########################################################################################################################################
################################################################ others ##################################################################
##########################################################################################################################################

#print with color
def cstr(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s)
def print_color(t):
    display(html_print(' '.join([cstr(ti, color=ci) for ti,ci in t])))

