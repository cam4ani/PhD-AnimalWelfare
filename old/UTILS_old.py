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

#plot
import matplotlib.pyplot as plt
import seaborn as sns

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
################################################ verification of configuration parameters ################################################
##########################################################################################################################################


def config_param_ver(config):
    
    #open param
    dico_zone_order = dico_zone_order.config
    dico_matching = dico_matching.config
    
    #test 1: the keys of dico_zone_order should be equal to some values of the dico_matching dictionary
    l = [x for x in dico_matching.values() if x not in dico_zone_order.keys()]
    if len(l)>0:
        print('ERROR: all values of dico_matching should have a key in the dico_zone_order ')
        print(l)
        sys.exit()
        
    #define zone and there order in the visual
    #verify all the values are unique (if not it should be changes in processing before hand, to know by which name we should be
    #refering too) 
    if len(set(dico_zone_order.values()))!=len(dico_zone_order.values()):
        print('dico_zone_order in config file should have unique values')
        sys.exit()        
        
    #TODO!!!

    return true
        
        
    
##########################################################################################################################################
############################################################ preprocessing ###############################################################
##########################################################################################################################################


def preprocessing_experiment2(paths, p_pen_info, config, save=True):
    
    '''each experiment should have his own function
    open from a list of csv-path all the csv and aggregated them and put into correct format
    output one df'''

    #initialise variables
    path_extracted_data = config.path_extracted_data
    id_run = config.id_run
    date_min = config.date_min
    date_max = config.date_max

    #create path to save extracted data/info if not existing
    if not os.path.exists(path_extracted_data):
        os.makedirs(path_extracted_data)

    #download all logs one by one adding the logfilename and the ts_order
    li_df = []
    for path in paths:
        log_name = path.split('initial_data_2experiment\\')[-1].split('.')[0].replace('\\','_').replace(' ','_')
        df = pd.read_csv(path, sep=';', names=['Timestamp', 'SerialNum', 'TrackingTag', 'Zone', 'Marker','U1','U2']) 
        df['log_file_name'] = log_name 
        df['ts_order'] = df.index.copy() 
        v = df.shape[0]
        if v<80000:
            print_color((('log: %s has '%log_name,'black'),(v,'red'),(' rows','black')))
        else:
            print_color((('log: %s has '%log_name,'black'),(v,'green'),(' rows','black')))
        li_df.append(df)
    df = pd.concat(li_df)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d.%m.%Y %H:%M:%S") #faster with specified format than parse_dates

    #open pen info, remove the rows with empty BackPackID value, and then say HenID is Pen+BackPackID
    df_pen_info = pd.read_excel(p_pen_info,sep=';',index_col=False) 
    df_pen_info = df_pen_info[~df_pen_info['BackPackID'].isnull()]
    df_pen_info['HenID'] = df_pen_info['Pen'].map(int).map(str)+df_pen_info['BackPackID']

    #some tracking tag can be associated to the same henID (e.g. 119, 101). From this dataframe we will create a dictionary with 
    #values being the TrackingTag and keys being the HenID names
    df['TrackingTag'] = df['TrackingTag'].map(str)
    df_pen_info['TrackingTag'] = df_pen_info['TrackingTag'].map(str)
    dico_TT_henID = dict(zip(df_pen_info['TrackingTag'], df_pen_info['HenID']))

    #match the tracking tag to the associated henid
    df['HenID'] = df['TrackingTag'].map(lambda x: dico_TT_henID[x] if x in dico_TT_henID else 'error')
    print_color((('There is ','black'),(df[df['HenID']=='error'].shape[0],'red'),
                 (' records with a TrackingTag(%s) that has no associated HenID, \
                 we will remove those records'%' '.join([str(x) for x in df[df['HenID']=='error']['TrackingTag'].unique()]),'black')))
    df = df[df['HenID']!='error']
    #df[df['HenID']=='10bb']['TrackingTag'].value_counts() #worked 101, 119
    df['PenID'] = df['HenID'].map(lambda x: re.search(r'\d+', x).group()) #first digits atching the string

    #make sure about the type
    df['PenID'] = df['PenID'].map(int).map(str)
    df['Zone'] = df['Zone'].map(lambda x: x.strip())
    df = df.sort_values(['Timestamp'], ascending=True)
    df = df.filter(['Timestamp', 'HenID', 'Zone','PenID','log_file_name','ts_order','TrackingTag']).reset_index(drop=True)

    #choose the date you want if any
    if date_min!=None:
        print(date_min)
        print('lets look at the record only between date %s and %s'%(str(date_min),str(date_max)))
        df = df[(df['Timestamp']>=date_min) & (df['Timestamp']<=date_max)]

    if save:
        df.to_csv(os.path.join(path_extracted_data, id_run+'_PreprocessRecords.csv'),sep=';')
    
    return(df)


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
    
    #Remove record associated to wrong HenID
    df['TagID_Pen_HenID'] = df['TagID_Pen_HenID'].map(lambda x: x.replace('_',' '))
    x0 = df[df['TagID_Pen_HenID']=='15C3'].shape[0]
    df = df[df['TagID_Pen_HenID']!='15C3']
    print('We remove %d records due to an unkown TagID_Pen_HenID value: 15C3, we are left with %d records'%(x0, df.shape[0]))
    df['HenID'] = df['TagID_Pen_HenID'].map(lambda x: x.split(' ')[-1])
    df['PenID'] = df['TagID_Pen_HenID'].map(lambda x: x.split(' ')[-1][0:-1].strip())
    df['Zone'] = df['Zone'].map(lambda x: x.strip())
    #TODO: remove if no need of ts_order
    df = df.sort_values(['Timestamp'], ascending=True)    
    #keep only usefull variables and return it
    df = df.filter(['Timestamp', 'HenID', 'Zone','PenID','log_file_name','Signal','ts_order']).reset_index(drop=True)


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
    
    #choose the date you want if any
    if date_min!=None:
        print(date_min)
        print('lets look at the record only between date %s and %s'%(str(date_min),str(date_max)))
        df = df[(df['Timestamp']>=date_min) & (df['Timestamp']<=date_max)]

    if save:
        df.to_csv(os.path.join(path_extracted_data, id_run+'_PreprocessRecords.csv'), sep=';')

    return(df)  



##########################################################################################################################################
############################################################# verification ###############################################################
##########################################################################################################################################


def verification_based_on_initial_record(df, config, min_daily_record_per_hen=500, last_hour_outside2inside=17, min_nbr_zone_per_day=5,
                                         min_nbr_boots_per_zone=5):
    
    '''This function will output some information that would allow to make daily verification of the systems, 
    on the last log file(s). 
    It will first output a series of table with inforamtives records
    Then it will clean the records regarding mainly flickering
    Finally it will generate time series and compute measure on those, to produce create more alerts'''
    
    #start recording the time it last
    START_TIME = time.clock()
    
    ###########################################################################################################################
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    print_color((('Aggregating the logfiles, adding pen info and look at basic info.........','blue'),))
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    
    #initialise variables
    path_extracted_data = config.path_extracted_data
    outside_zone = config.outside_zone
    dico_matching = config.dico_matching
    
    #change zone name
    df['Zone'] = df['Zone'].map(lambda x: dico_matching[x])
    
    #pen info
    df['day'] = df['Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day)) 
    print_color((('Number of daily record in each Zone','blue'),))
    df_ = df.groupby(['day','Zone'])['Timestamp'].count().reset_index().sort_values(['Timestamp'])
    df_.rename(columns={'Timestamp':'nbr of records'}, inplace=True)
    display(df_.groupby('day')['Zone','nbr of records'].agg(lambda x: tuple(x)))
    
    #hen info
    print_color((('Number of daily record for each hen that has less than %d records'%min_daily_record_per_hen,'blue'),))
    #display(df['HenID'].value_counts())    
    df_ = df.groupby(['day','HenID'])['Timestamp'].count().reset_index().sort_values(['Timestamp'])
    df_.rename(columns={'Timestamp':'nbr of records'}, inplace=True)
    display(df_[df_['nbr of records']<=min_daily_record_per_hen].groupby('day')['HenID',
                                                                                'nbr of records'].agg(lambda x: tuple(x)))    

    
    ###########################################################################################################################
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    print_color((('Timestamp info.........','blue'),))
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    
    #Context: informing on the first record date of each log (in case we are in the middle of a day)
    #also useful when you are searchiing to verify some recors, with this you know which log file to open 
    print_color((('Timestamp of first record for each log files','blue'),))
    display(df.groupby(['log_file_name'])['Timestamp'].agg(lambda x: min(x)).reset_index())
    print('')

    #(no need, included in next print) timestamp of first record for each day (not each hen as otherwise to much info)
    #print_color((('Timestamp of first record for each day','blue'),))
    #display(df.groupby(['day'])['Timestamp'].agg(lambda x: min(list(x))).reset_index())   
    
    #more precisely for each zone (not each hen as otherwise to much info)
    print_color((('Timestamp of first record for each day in each zone','blue'),))
    df_ = df.groupby(['day','Zone'])['Timestamp'].agg(lambda x: min(list(x))).reset_index().sort_values(['Timestamp'])
    df_.rename(columns={'Timestamp':'first timestamp'}, inplace=True)
    display(df_.groupby('day')['Zone','first timestamp'].agg(lambda x: tuple(x)) )  
    
    ###########################################################################################################################
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    print_color((('Hen info.........','blue'),))
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))

    #last time each hen went back from outside (verify when outside close and if hen slept outside)
    if outside_zone!=None:
        print_color((('All last time a hen went back from outside later than %d hour of the same day (i.e. could by any time \
        the day after too)'%int(last_hour_outside2inside), 'blue'),))
        #add previous zone variable
        li_df = []
        for i, df_hen in df.groupby(['HenID']):
            #as the next record date (sort by date, then simply shift by one row and add nan at then end)
            df_hen = df_hen.sort_values(['Timestamp'], ascending=True) #ts_order
            df_hen['previous_record_date'] = [np.nan]+df_hen['Timestamp'].tolist()[0:-1]
            df_hen['outsidezone record day'] = [np.nan]+df_hen['day'].tolist()[0:-1]
            df_hen['previous_zone'] = [np.nan]+df_hen['Zone'].tolist()[0:-1]
            li_df.append(df_hen)
        df__ = pd.concat(li_df)
        df_ = df__[df__['previous_zone']==outside_zone].groupby(['outsidezone record day','HenID'])['Timestamp'].agg(lambda x: max(x)).reset_index()
        df_.rename(columns={'Timestamp':'Timestamp record after outsidezone'}, inplace=True)
        #df_['outsidezone record day'] = df_['outsidezone record day'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df_['last timestamp too late'] = df_.apply(lambda x: x['Timestamp record after outsidezone']>=dt.datetime(x['outsidezone record day'].year,x['outsidezone record day'].month,x['outsidezone record day'].day,last_hour_outside2inside,0,0), axis=1)
        display(df_[df_['last timestamp too late']].drop(['last timestamp too late'],axis=1))
        #df_['hour'] = df_['Timestamp record after outsidezone'].map(lambda x: x.hour)
        #display(df_[df_['hour']>=last_hour_outside2inside].drop(['hour'],axis=1))
    else:
        print('no outside zone in config file, will not check for last time a hen went back from outside later')
        
    #each hen goes every day in each zone? how often?
    print_color((('All event where a hen has less (or equal) than %d bouts in a zone on a day, or has went in less (or equal) \
    than %d zone on a day'%(int(min_nbr_boots_per_zone), int(min_nbr_zone_per_day)),'blue'),))
    df_ = df.groupby(['day','HenID','Zone'])['Timestamp'].count().reset_index().sort_values(['Timestamp'])
    df_.rename(columns={'Timestamp':'nbr of record'}, inplace=True)
    df_ = df_.groupby(['day','HenID'])['Zone','nbr of record'].agg(lambda x: tuple(x)).reset_index()
    df_['nbr zone went too'] = df_['Zone'].map(lambda x: len(x))
    df_['min nbr of bouts in a zone'] = df_['nbr of record'].map(lambda x: min(x))
    display(df_[(df_['nbr zone went too']<=min_nbr_zone_per_day)|(df_['min nbr of bouts in a zone']<=min_nbr_boots_per_zone)])
    
    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))
    

        

##########################################################################################################################################
############################################################### cleaning #################################################################
##########################################################################################################################################

def zone_sequence(li, nbrZone=2):
    '''from a list it will output a list of the maximum number of last entries until reaching a maximum of nbrZOne different element'''
    r = []
    k = len(li)-1
    while len(set(r+[li[k]]))<=nbrZone:
        v = li[k]
        r.append(v)
        k = k-1
        #print(k,r)
        if k==-1:
            break
    return(list(reversed(r)))
#small ex
#print(zone_sequence([1,2,1,2,1,2,2,2,3,1,1],2)) #[3, 1, 1]

def zone_sequence_sens2(li, nbrZone=2, li_t=None):
    '''from a list it will output a list of the maximum number of last entries until reaching a maximum of nbrZOne different element'''
    r = []
    t = []
    k = 0
    while len(set(r+[li[k]]))<=nbrZone:
        v = li[k]
        r.append(v)
        if li_t!=None:
            vt = li_t[k]
            t.append(vt)
        k = k+1
        #print(k,r)
        if k==len(li):
            break
    return(r,t)
#small ex
#zone_sequence_sens2([1,2,1,2,1,2,2,2,3,1,1],3,[1,2,3,4,5,6,7,8,9,10,11])
#-->([1, 2, 1, 2, 1, 2, 2, 2, 3, 1, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
#zone_sequence_sens2([1,2,1,2,1,2,2,2,3,1,1],2,[1,2,3,4,5,6,7,8,9,10,11])
#-->([1, 2, 1, 2, 1, 2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8])
#zone_sequence_sens2([1,2,1,2,1,2,2,2,3,1,1],2)
#-->([1, 2, 1, 2, 1, 2, 2, 2], [])



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
    li_colnames = ['Timestamp','Zone','HenID','PenID','log_file_name','ts_order']
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
    df['day'] = df['Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day))
    
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
        #add thets_order_logname associated to that record so that it appears 2 times
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
    df = df.filter(['Timestamp','HenID','Zone','PenID','log_file_name','day','ts_order_logname','ts_order_list','ms',
                     'Timestamp_initial'])
    if save:
        df.to_csv(os.path.join(path_extracted_data, id_run+'_records_GeneralCleaning.csv'), sep=';', index=False) 
        
    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))  
    
    return(df)



def simple_cleaning_experiment2(df_ts, config, nbr_sec_mean, mi=None, ma=None, save=True):
    
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
    nbr_sec = config.nbr_sec
        
    #######################################################################################################################
    #create a list of dates that we want starting from our initial and end dates
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
    #add new timestamp and groupby
    df_date = pd.DataFrame({'New_Timestamp':Daterange})
    df_date['New_Timestamp'] = df_date['New_Timestamp'].map(lambda x: pd.to_datetime(x))
    df_ts = pd.merge_asof(df_ts, df_date, left_on=['Timestamp'], right_on=['New_Timestamp'], direction='forward')
    #df_ts['New_Timestamp_old'] = df_ts['Timestamp'].map(lambda x: min([d for d in Daterange if d >= x], default=np.nan))
    
    #create a director if not existing
    path_ = os.path.join(path_extracted_data, 'HensTimeSeries')
    if not os.path.exists(os.path.join(path_)):
        os.makedirs(os.path.join(path_))
        
    #for each hen take its time serie and find the most frequent zone per new_timestamp, and save if
    for h in tqdm.tqdm([x for x in df_ts.columns if x.startswith('hen_')]):
        df_ = df_ts[[h,'New_Timestamp']].copy()
        df_['nbr_sec'] = nbr_sec
        #df_verification = df_ts.groupby(['New_Timestamp']).agg({'Timestamp':['max', 'min']}).reset_index()
        #df_verification.to_csv(os.path.join(path_ ,id_run+'_ts_MostFrequentZone_period_VERIFICATION'+str(nbr_sec_mean)+'_'+str(mi).split(' ')[0]+'_'+str(ma).split(' ')[0]+'_'+h+'.csv'), sep=';', index=False)
        df__ = df_.groupby(['New_Timestamp',h])['nbr_sec'].sum().reset_index() #count as we have seconds
        df_final = df__.groupby(['New_Timestamp'])[h,'nbr_sec'].agg(lambda x: tuple(x)).reset_index()
        df_final['most_frequent_zone'] = df_final.apply(lambda x: x[h][x['nbr_sec'].index(max(x['nbr_sec']))], axis=1)
        df_final['nbr_duration_per_zone'] = df_final.apply(lambda x: str({x[h][k]:x['nbr_sec'][k] for k in range(len(x[h]))}), axis=1)
        df_final['nbr_lost_duration_per_zone'] = df_final['nbr_duration_per_zone'].map(lambda x: str({z:v for z,v in eval(x).items() if \
                                                                           v!=max(eval(x).values())}))
        df_final['nbr_lost_duration'] = df_final['nbr_lost_duration_per_zone'].map(lambda x: sum(eval(x).values()))
        df_final['perc_lost_duration'] = df_final['nbr_lost_duration'].map(lambda x: x/nbr_sec_mean*100)
        df_final['day'] = df_final['New_Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day))
        if save:
            df_final.to_csv(os.path.join(path_ ,id_run+'_ts_MostFrequentZone_period'+str(nbr_sec_mean)+'_'+str(mi).split(' ')[0]+\
                                         '_'+str(ma).split(' ')[0]+'_'+h+'.csv'), sep=';', index=False)
    
    #running time info and return final cleaned df
    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))
    
    return 


def cleaning_mouvement_records(df, config, nbr_block_repetition, flickering_type1=True, save=True, is_bb_experiment=False,
                               interzone_name=True):
    
    #start recording the time it last
    START_TIME = time.clock()
    
    print_color((('We Start with ','black'),(df.shape[0],'green'),(' initial records','black')))    
    #initialize parameters
    id_run = config.id_run
    path_extracted_data = config.path_extracted_data
    li_date2remove = config.li_date2remove
    nbrZone = config.nbrZone
    nbr_sec_flickering1 = config.nbr_sec_flickering1
    #create path if it does not exist
    if not os.path.exists(path_extracted_data):
        os.makedirs(path_extracted_data)
    dico_matching = config.dico_matching
    dico_impossible_mvt_to = config.dico_impossible_mvt_to
    li_not_flickering2 = config.li_not_flickering2
    dico_night_hour = config.dico_night_hour    
    
    
    #######################################################################################################################
    ################# handle flickering type2 : flickering that we know should not exist, and we could correct
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    print_color((('Handle flickering type2.........','blue'),))
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    #Note that it will be done before the usual flickering situations, as we
    #in case of bb data: for "nbr_block_repetition" consecutive sequences of Bi-Zi for any i and Z, no matter if the nbr_sec_flickering1
    #is bigger than 3 sec, then it should be called interzone_BR (or Box, ask yamenah for final decision). If this happens in middle of
    #flickering situation and the naming of one record would be different depenidn on flickeirng type 1 ou flickering type 2, we decided
    #that its the type 2 that should be more powerful. Hence, we start with this and then continue with flickering type1 on this value,
    #and wont include the zone named with already with interzone. Rule: should start and end with box, box should be the same
    li_df = []
    for k, df_hen in tqdm.tqdm(df.groupby(['HenID'])):
        #TODO: remove if no need of ts_order
        df_hen = df_hen.sort_values(['Timestamp'], ascending=True)
        df_hen['Zone_sequence'] = df_hen['Zone'].map(lambda x: [x[0]])
        df_hen['Zone_sequence_timestamp'] = df_hen['Timestamp'].map(lambda x: [x])
        #note that one record can be the first and the last of a sequence (i.e. not clear which one to remove)
        df_hen['test_isfirst_flickering2'] = False
        df_hen['test_islast_flickering2'] = False
        df_hen['test_is_flickering2'] = False
        df_hen = df_hen.reset_index(drop=True) #as index might have wholes
        for i in reversed(range(0,df_hen.shape[0]-1)):
            x0 = df_hen.iloc[i].copy()
            x1 = df_hen.iloc[i+1].copy() #add set for efficiency and if we dont need all info of the sequence
            x0t = x0['Zone_sequence_timestamp']
            x0s = x0['Zone_sequence']
            x0s.extend(x1['Zone_sequence'])
            x0t.extend(x1['Zone_sequence_timestamp'])
            newval, newt = zone_sequence_sens2(x0s, nbrZone, x0t)
            #not recording as sequence when the sequence is in li_not_flickering2 parameter (ER can happen with more than 4000 
            #records making things note fficient for soemthing we dont care as its normal)
            if set(newval) in li_not_flickering2:
                #if its not a possible flickering type2 sequence, then we will keep only the last records representing one zone
                #e.g. ererrreeerererereeeeeeebebeeebebebbbebe : we should keep the eeee sequence in the br sequence even if it also
                #belong to the er non-flickering sequence 
                newval, newt = zone_sequence_sens2(x0s, nbrZone-1, x0t) #if nbrZone!=2 then must change this line
            df_hen.at[i, 'Zone_sequence'] = newval
            df_hen.at[i, 'Zone_sequence_timestamp'] = newt
            #if the list is smaller or equal (i.e. we still have remove one element and took the new one), then it means that the 
            #next record is the first record of a sequence.
            if (len(newval)<len(x0s)) & (i!=(df_hen.shape[0]-2)):
                df_hen.at[i+1,'test_isfirst_flickering2'] = True
            #Or if its the first record, then its necessary also the first record of a sequence
            if i==0:
                df_hen.at[i, 'test_isfirst_flickering2'] = True
            #OR we could have said: is first record of a flickering type 2 event if the previous sequence type is different than the
            #actual sequence type. But this was not used as we would need to go again in a for loop
        #remove first_flickering2 record when they belong to unwanted sequence (i.e. normal sequence, i.e. in li_not_flickering2) 
        #so that we wont search for end of sequence of these unwanted sequence
        df_hen['nbr_record'] = df_hen['Zone_sequence'].map(lambda x: len(sequence_without_cez(x)))
        df_hen['nbr_record_with_cez'] = df_hen['Zone_sequence'].map(lambda x: len(x))
        #when it belongs to a non-flickeringtype2 sequence (i.e. defined by li_not_flickering2) then does not count as a sequence
        df_hen.loc[df_hen['nbr_record']==1, 'test_isfirst_flickering2'] = False
        #the last entry of a sequence can be defined by looking at all first entry of a sequence and subtracting to the row index the 
        #number of record there is in the sequence
        #we need a list and not a zip as we will use it twice and zip can only be used once
        li_index = list(zip(df_hen[df_hen['test_isfirst_flickering2']].index, 
                            df_hen[df_hen['test_isfirst_flickering2']]['nbr_record'],
                            df_hen[df_hen['test_isfirst_flickering2']]['nbr_record_with_cez']))
        li_index_last = [x[0]+x[2]-1 for x in li_index]
        #add all index not only the last one
        li_index_all = [x[0]+i for x in li_index for i in range(0,x[2]) if x[1]>=nbr_block_repetition*nbrZone] 
        df_hen.loc[li_index_last, 'test_islast_flickering2'] = True
        df_hen.loc[li_index_all, 'test_is_flickering2'] = True
        li_df.append(df_hen)
    #put again in one dataframe
    df = pd.concat(li_df)

    #add details
    df['Zone_sequence_without_cez'] = df['Zone_sequence'].map(lambda x: sequence_without_cez(x))
    df['sequence_at_least_x_repetition'] = df['nbr_record'].map(lambda x: True if x>=nbr_block_repetition*nbrZone else False)
    df['sequence_type'] = df['Zone_sequence_without_cez'].map(lambda x: ''.join(sorted([x for x in set(x)])))
    df['zone_flickering2'] = np.where(df['test_is_flickering2']==False, df['Zone'], 'flickering2')

    #save csv for verification
    for s in df[df['sequence_at_least_x_repetition']]['sequence_type'].unique():

        #choose the first record of each event that has at least x repetitions and that are of type s
        df_ = df[(df['sequence_at_least_x_repetition']) & (df['test_isfirst_flickering2']) \
                 & (df['sequence_type']==s)][['HenID','log_file_name','Zone_sequence_timestamp','Zone_sequence',
                                              'Zone_sequence_without_cez','nbr_record']].copy()

        #add variable at event level
        df_['duration_btw2zone'] = df_['Zone_sequence_timestamp'].map(lambda x: [(x[k+1]-x[k]).seconds for k in range(len(x)-1)])
        #lets remove durations that might be biaised by sleeping
        #if the 2nd timestamp is later than the last possible daily-timestamp of the day of the first tiemstamp, then we should put -1
        #But, not that if the first timestamp is already happaning during the night, then it wont work, so we will use this only when 
        #both timestamp are hapening during the day, and otherwise we will also put -1:
        #Put -1 if at least one of the 2 timestamp happens during the night or if none happend during the night but the 2nd timestamp
        #is later than the last possible daily-timestamp of the day of the first timestamp
        df_['day_duration_btw2zone'] = df_['Zone_sequence_timestamp'].map(lambda x: [day_duration(x[k], x[k+1],
                                                                                          dico_night_hour) for k in range(len(x)-1)])
        df_['max_dayduration_btw2zones_sec'] = df_['day_duration_btw2zone'].map(lambda x:max(x))

        #add condition that it must begin and end with box
        if is_bb_experiment:
            df_['begin_and_end_with_box'] = df_['Zone_sequence_without_cez'].map(lambda x: len(''.join(x).strip('R').strip('E'))>=3* nbrZone)

        df_.sort_values('max_dayduration_btw2zones_sec', ascending=False, inplace=True)
        if save:
            df_.to_csv(os.path.join(path_extracted_data, id_run+'_'+s+'_event_'+str(nbr_block_repetition)+'.csv'), sep=';', index=False)

        #groupby hen into lists and add variable at hen level
        df_ = df_.groupby(['HenID'])['log_file_name','Zone_sequence_timestamp','Zone_sequence_without_cez','Zone_sequence',
                                     'nbr_record','duration_btw2zone','day_duration_btw2zone',
                                     'max_dayduration_btw2zones_sec'].agg(lambda x: list(x)).reset_index() 
        df_['nbr of such event'] = df_['log_file_name'].map(lambda x: len(x))
        df_['overall_max_nbr_record_in_one_event'] = df_['nbr_record'].map(lambda x: max(x))
        df_.rename({'nbr_record':'nbr of record (without counting consecutives equal)'}, inplace=True, axis=1)
        df_['overall_max_dayduration_btw2zones_sec'] = df_['max_dayduration_btw2zones_sec'].map(lambda x: max(x) if len(x)>0 else 0)

        df_.sort_values('nbr of such event', ascending=False, inplace=True)
        if save:
            df_.to_csv(os.path.join(path_extracted_data, id_run+'_'+s+'_event_hen_level_all_info_'+str(nbr_block_repetition)+'.csv'),
                       sep=';', index=False)
            df_.filter(['HenID', 'nbr of record (without counting consecutives equal)','max_dayduration_btw2zones_sec',
                        'nbr of such event','overall_max_nbr_record_in_one_event','overall_max_dayduration_btw2zones_sec'], 
                       axis=1).to_csv(os.path.join(path_extracted_data,
                                                   id_run+'_'+s+'_event_hen_level_'+str(nbr_block_repetition)+'.csv'), 
                                      sep=';', index=False)

        #print info on maximum duration between two blocks
        m = df_['overall_max_dayduration_btw2zones_sec'].max()
        print('-----------------------------------------------------------------------------------')
        if m==0:
            print('no consecutives \"%s\" block event'%''.join(li))
        else:
            x = df_[df_['overall_max_dayduration_btw2zones_sec']==m]
            print('The max duration between 2 zones (during day) in event type \"%s\" is %.2f (mn), e.g. %s:'%(s, m/60,
                                                                                                               x['HenID'].values[0]))
            ind = x['max_dayduration_btw2zones_sec'].values[0].index(m)
            display(x['Zone_sequence_timestamp'].values[0][ind])
            #info on the maximum number of records
            m = df_['overall_max_nbr_record_in_one_event'].max()
            x = df_[df_['overall_max_nbr_record_in_one_event']==m]
            ind = x['nbr of record (without counting consecutives equal)'].values[0].index(m)
            li_dates = x['Zone_sequence_timestamp'].values[0][ind]
            print('The maximum number of records (without counting consecutives equal) is %d'%m)
            print('e.g. %s, starting at %s end ending at %s'%(x['HenID'].values[0], str(min(li_dates)), str(max(li_dates))))  

    #save plot for verification
    s_t = {'HenID':3, 'PenID':10} #size of x label column
    path_ =  os.path.join(path_extracted_data,'visual','for_verification')
    #create a director if not existing
    if not os.path.exists(path_):
        os.makedirs(path_)
    for type_ in ['HenID','PenID']:
        for s in df[df['sequence_at_least_x_repetition']]['sequence_type'].unique():
            df_ = df[(df['sequence_at_least_x_repetition']) & (df['test_isfirst_flickering2']) &(df['sequence_type']==s)]
            df_plot = df_.groupby([type_])['log_file_name'].count().reset_index().sort_values('log_file_name', ascending=False)
            x = df_plot[type_].tolist()
            y = df_plot['log_file_name'].tolist()
            fig = plt.figure()
            ax = plt.subplot(111)
            width = 0.8
            ax.bar(range(len(x)), y, width=width)
            ax.set_xticks(np.arange(len(x)) + width/2)
            ax.set_xticklabels(x, rotation=90, size=s_t[type_]);
            plt.title('Event nbr of consecutives %s block (min %d)'%(s, nbr_block_repetition))
            plt.xlabel(type_)
            if save:
                plt.savefig(os.path.join(path_, id_run+'_'+type_+'_event_nbr_consecutives_'+s+'_block.png'), dpi=300,
                            format='png', bbox_inches='tight')
            plt.show()
            plt.close()

    #TODO: what should we do with it??


    #######################################################################################################################
    #ceiz = consecutives equal initial zone
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    print_color((('Compute variable for flickering type1, impossible movement and ceiz.........','blue'),))
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))

    li_df = []
    #more efficient to do it per hen, as it wont need to search in the whole dataframe, and we can simply shift the timestamp column
    for i, df_hen in tqdm.tqdm(df.groupby(['HenID'])):
        #as the next record date (sort by date, then simply shift by one row and add nan at then end)
        #TODO: remove if no need of ts_order
        df_hen = df_hen.sort_values(['Timestamp'], ascending=True) #ts_order
        #same date, one must take the last recorded one & sorting by date might change it. Also it already should be sorted by date
        df_hen['next_record_date'] = df_hen['Timestamp'].tolist()[1:]+[np.nan]
        #compute duration
        df_hen['duration'] = df_hen.apply(lambda x: x['next_record_date']-x['Timestamp'], axis=1)
        #compute the last record date in order to put interzone also when the duration is >=nbr_sec_flickering1
        df_hen['previous_record_date'] = [np.nan]+df_hen['Timestamp'].tolist()[0:-1]
        #compute previous duration in order to put interzone also when the duration is >=nbr_sec_flickering1
        df_hen['previous_duration'] = [np.nan]+df_hen['duration'].tolist()[0:-1]
        #add next record for the impossible movement
        df_hen['next_zone'] = df_hen['Zone'].tolist()[1:]+[np.nan]
        #add previous record for the consecutives equal initial zones
        df_hen['previous_zone'] = [np.nan]+df_hen['Zone'].tolist()[0:-1]
        df_hen['previous_previous_zone'] = [np.nan]+df_hen['previous_zone'].tolist()[0:-1]
        li_df.append(df_hen)
    #put again in one dataframe
    df = pd.concat(li_df)
    #dont care about the false positive warning   

    #######################################################################################################################
    ################# handle flickering situations
    #A flickering situation happens when a hen change zone within strictly less than 2seconds, in which case we name these 
    #situations "Interzone" and keep only the first timestamp of each interzones situation
    if flickering_type1:
        print_color((('-----------------------------------------------------------------------------------------------','blue'),))
        print_color((('Handle flickering type1.........','blue'),))
        print_color((('-----------------------------------------------------------------------------------------------','blue'),))

        ######## name interzone and interzone_f
        #interzone when duration is less than 2 seconds
        #note that there is no need to merge interzone in one timestamp as we will in any case extend to a time serie for analysis
        df['test_Zone_without_flickering_nonaming'] = df['Zone'].copy()
        df.loc[df['duration']<dt.timedelta(seconds=nbr_sec_flickering1),'test_Zone_without_flickering_nonaming'] = 'Interzone'
        #interzone_f (i.e. end of interzone situation) if its not interzone (i.e. its duration is longer than 3 seconds) and its
        #previous duration is shorter than 3 seconds
        df.loc[(df['previous_duration']<dt.timedelta(seconds=nbr_sec_flickering1))&\
               (df['test_Zone_without_flickering_nonaming']!='Interzone'),
               'test_Zone_without_flickering_nonaming'] = 'Interzone_f'
        #we wont be doing this, as otherwise if the last timestamp are flickering situation, we will miss a zone
        #replace 'test_Zone_without_flickering_nonaming' by np.nan if the duration is nan (i.e. if last observation)
        #df.loc[pd.isnull(df['duration']),'test_Zone_without_flickering_nonaming'] = np.nan

        ######## flag remove-flickeringtype1 the zone that are interzone_f or the one that are interzone with a previous zone also 
        #named interzone
        print('Lets flag the zone to remove due to flickering type 1')
        li_df = []
        for i, df_hen in tqdm.tqdm(df.groupby(['HenID'])):
            #as the next record date (sort by date, then simply shift by one row and add nan at then end)
            #TODO: remove if no need of ts_order
            df_hen = df_hen.sort_values(['Timestamp'], ascending=True) #ts_order
            df_hen['previous_test_Zone_without_flickering_nonaming'] = [np.nan]+\
            df_hen['test_Zone_without_flickering_nonaming'].tolist()[0:-1]
            li_df.append(df_hen)
        #put again in one dataframe
        df = pd.concat(li_df)
        df['test_tuple_previousinter_inter'] = list(zip(df['previous_test_Zone_without_flickering_nonaming'],
                                                        df['test_Zone_without_flickering_nonaming']))
        df['test_ToRemove_flickering1'] = df['test_tuple_previousinter_inter'].map(lambda x: True if ((x[1]=='Interzone_f') | \
                                                                                   ((x[0]=='Interzone')&(x[1]=='Interzone'))) else False)

        ######## name the interzone according to all zone associated to the flickering event
        df['Zone_without_flickering'] = df['test_Zone_without_flickering_nonaming'].copy()
        if interzone_name:
            print('Lets give interzone some names. This is long as we need to look at the next zone only once we have changed the \
            next row') 
            li_df = []
            for k, df_hen in tqdm.tqdm(df.groupby(['HenID'])):
                #TODO: remove if no need of ts_order
                df_hen = df_hen.sort_values(['Timestamp'], ascending=True)
                df_hen['interzone_info'] = df_hen['Zone'].copy()
                df_hen = df_hen.reset_index(drop=True)
                #as we will keep the first entry of consecutives equal zones, we will take the value from the next record to the previous 
                #one (and not the opposite way) (i.e. start from the end of dataframe)
                #idea: if the next zone is interzone_f then put the info of now and next. If next and actual zones are interzone, 
                #then put the info of now and next. Otherwise put nothing
                for i in reversed(range(0,df_hen.shape[0]-1)):
                    x0 = df_hen.iloc[i].copy()
                    x1 = df_hen.iloc[i+1].copy()
                    if (x1['test_Zone_without_flickering_nonaming']=='Interzone_f') | \
                    (x0['test_Zone_without_flickering_nonaming']=='Interzone') & \
                    (x1['test_Zone_without_flickering_nonaming']=='Interzone'):
                        df_hen.at[i,'interzone_info'] = x0['interzone_info']+','+x1['interzone_info']
                    else: 
                        #cant put '' as otherwise we will miss the zone of the last record of an event
                        df_hen.at[i,'interzone_info'] = x0['interzone_info'] 
                li_df.append(df_hen)
            #put again in one dataframe
            df = pd.concat(li_df)
            #dont care about the false positive warning

            #define unique naming based on the flickering information
            df['interzone_name'] = df['interzone_info'].map(lambda x: 'Interzone_'+\
                                   ''.join(sorted([str(j[0]) for j in set([i.strip() for i in x.split(',') if len(i)>0])])) )
            df['Zone_without_flickering'] = np.where(df['Zone_without_flickering']=='Interzone', 
                                                     df['interzone_name'], 
                                                     df['Zone_without_flickering'])

        #replace interzone of only one type of zone by the zone name (i.e. its actually cez, which shouldnt be #flickering
        dico_interzoneCorrection = {z:[i for i in dico_matching.values() if i.startswith(z.split('_')[1])][0] for z in\
                                [x for x in df['Zone_without_flickering'].unique() if x.startswith('Interzone_')]  if \
                                    (len(z.split('_')[1])==1)&(z.split('_')[1]!='f')}
        print('dictionary for one zone flickering situation: ',dico_interzoneCorrection)
        df['Zone_without_flickering'] = df['Zone_without_flickering'].map(lambda x: dico_interzoneCorrection[x] if x in \
                                                                          dico_interzoneCorrection else x)
        print('possible \"Zone_without_flickering\" values: ', df['Zone_without_flickering'].unique())


    #######################################################################################################################
    ################# remove impossible movement
    #add next zone based on Zone_without_flickering (for quality verification)
    #note that removing consecutives equal zone must be done after defining interzones, as otherwise, removing these consecutives 
    #zone might break a flickering situation in two.
    #note that consecutives equal zone that are flickering too already hav been removed in the flickering of type 2
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    print_color((('Find impossible movement not in flickering type 2.........','blue'),))
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))

    if dico_impossible_mvt_to==None:
        print('No impossible movement defined, we will skeep this step')
    else:
        df['next_zone'].fillna('no_next_zone', inplace=True)
        df['test_tuple_record_nextrecord'] = list(zip(df['Zone'], df['next_zone']))
        df['mvt_type'] = df['test_tuple_record_nextrecord'].map(lambda x: str(x[0][0])+str(x[1][0]))
        #check if all zones are in the dico_impossible_mvt_to
        if any([i for i in [v[0] for v in df['Zone'].unique()] if i not in dico_impossible_mvt_to.keys()]):
            print('your \"dico_impossible_mvt_to\" parameter does not have all possible zones')
            print('these are the zone we need: ', [v[0] for v in df['Zone'].unique()])
            sys.exit()
        df['is_impossible_mvt'] = df['mvt_type'].map(lambda x: x[1] in dico_impossible_mvt_to[x[0]])

        #save plot and csv for impossible movemnt that is not in flickering
        li_mvt_type = df[(~df['test_is_flickering2']) & (df['is_impossible_mvt'])]['mvt_type'].unique()    
        s_t = {'HenID':3, 'PenID':10} #size of xlabel
        path_ =  os.path.join(path_extracted_data,'visual','for_verification')
        for s in li_mvt_type:
            df__ = df[(~df['test_is_flickering2']) & (df['is_impossible_mvt']) & (df['mvt_type']==s)].copy()

            #save
            df_ = df__.groupby(['HenID'])['Timestamp','log_file_name'].agg(lambda x: list(x)).reset_index()
            df_['nbr_impossible_mvt'] = df_['Timestamp'].map(lambda x: len(x))
            df_.sort_values(['nbr_impossible_mvt'], ascending=False, inplace=True)
            df_.rename({'Timestamp':'first timestamp of impossible mvt %s'%s}, axis=1, inplace=True)
            df_.to_csv(os.path.join(path_extracted_data, id_run+'_'+s+'_impossible_mvt_'+str(nbr_block_repetition)+'.csv'),sep=';',
                       index=False)

            #plot
            for type_ in ['HenID','PenID']:    
                df_plot = df__.groupby([type_])['log_file_name'].count().reset_index().sort_values('log_file_name', ascending=False)
                x = df_plot[type_].tolist()
                y = df_plot['log_file_name'].tolist()
                fig = plt.figure()
                ax = plt.subplot(111)
                width = 0.8
                ax.bar(range(len(x)), y, width=width)
                ax.set_xticks(np.arange(len(x)) + width/2)
                ax.set_xticklabels(x, rotation=90, size=s_t[type_]);
                plt.title('Nbr of impossible mouvement %s'%s)
                plt.xlabel(type_)
                plt.savefig(os.path.join(path_,id_run+'_'+type_+'_nbr_impossible_mvt_'+s+'.png'),dpi=300,
                            format='png',bbox_inches='tight')
                plt.show()
                plt.close()
    #TODO: what should we do with it??      

    
    #######################################################################################################################
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    print_color((('Finding the enveloppe situations.........','blue'),))
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))

    df['is_end_enveloppe']= False
    df.loc[(df['previous_zone']!=df['Zone']) & (df['previous_previous_zone']==df['Zone']) \
                               &(df['previous_duration']<=dt.timedelta(seconds=5)),'is_end_enveloppe'] = True

    li_df = []
    for i, df_hen in tqdm.tqdm(df.groupby(['HenID'])):
        df_hen = df_hen.sort_values(['Timestamp'], ascending=True)
        df_hen = df_hen.reset_index(drop=True) #as index might have wholes
        li_index_envelop = [x-i for x in df_hen[df_hen['is_end_enveloppe']].index for i in range(0,3)]
        df_hen['is_enveloppe'] = False
        df_hen.loc[li_index_envelop, 'is_enveloppe'] = True        
        li_df.append(df_hen)
    #put again in one dataframe
    df = pd.concat(li_df)
    #dont care about the false positive warning      

    df['enveloppe'] = np.where(df['is_enveloppe']==True, 'enveloppe', df['Zone'])


    #######################################################################################################################
    ################# remove consecutives equal Zone for same hens 
    #add next zone based on Zone_without_flickering (for quality verification)
    #note that removing consecutives equal zone must be done after defining interzones, as otherwise, removing these consecutives 
    #zone might break a flickering situation in two.
    #note that consecutives equal zone that are flickering too already hav been removed in the flickering of type 2
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    print_color((('Remove consecutives equal initial Zone for same hens.........','blue'),))
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))

    #True if next zone is equal to the actual zone
    df['correction_is_consecutive_equal_initial_zone'] = False
    #if the previous zone is the same, then its cez that should be removed
    df.loc[df['previous_zone']==df['Zone'], 'correction_is_consecutive_equal_initial_zone'] = True


    #######################################################################################################################
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    print_color((('Remove the consecutive equal final zone (e.g. two consecutives same flickering but distinct event).........','blue'),))
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))

    #TODO
    #to compute: previous_zonewf
    #perhaps needed: df['Zone_without_flickering'] = df['Zone_without_flickering'].fillna('')
    #df.loc[df['previous_zonewf']==df['Zone_without_flickering'], 'correction_is_consecutive_equal_zone'] = True


    #######################################################################################################################
    ################# Remove dates with healthcare
    #remove the healthassement days at the end of the cleaning, so that all the other record are cleaned accordingly and we assume its
    #correct
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    print_color((('Remove dates with health care.........','blue'),))
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))

    df['test_toberemoved_date'] = df['day'].isin(li_date2remove)
    x0 = df.shape[0]
    df = df[~df['test_toberemoved_date']]
    print_color((('We have ','black'),(df.shape[0],'green'),(' records (','black'),(x0-df.shape[0],'red'),
                 (' removed due to health-assessment dates)','black')))

    #######################################################################################################################
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))
    print_color((('Lets save a record file with all info and one without wrong records........','blue'),))
    print_color((('-----------------------------------------------------------------------------------------------','blue'),))

    #TODO WHEN WE KNOW WHAT TODO
    if save:
        t1 = time.clock()
        df.to_csv(os.path.join(path_extracted_data,id_run+'_record_with_allcleaned_info_'+str(nbr_block_repetition)+'.csv'), 
                  index=False, sep=';')

        li_info = ['log_file_name', 'PenID', 'Timestamp', 'HenID', 'Zone', 'Zone_without_flickering', 'test_ToRemove_flickering1',
                   'test_isfirst_flickering2', 'sequence_type', 'is_impossible_mvt', 'mvt_type','sequence_at_least_x_repetition',
                   'zone_flickering2','enveloppe']
        #remove consecutives equal initital zones
        #TODO: More efficient!!!
        df_info = df[~(df['correction_is_consecutive_equal_initial_zone'])][li_info].copy()
        df_info['Zone_without_flickering'] = df_info.apply(lambda x: '' if x['test_ToRemove_flickering1'] \
                                                           else x['Zone_without_flickering'], axis=1)
        df_info['flickering2_name'] = df_info.apply(lambda x: x['sequence_type'] if x['test_isfirst_flickering2'] & \
                                                    x['sequence_at_least_x_repetition'] else '', axis=1)
        df_info['is_impossible_mvt'] = df_info.apply(lambda x: x['mvt_type'] if x['is_impossible_mvt'] else '', axis=1)
        df_info.drop(['test_ToRemove_flickering1','sequence_type','test_isfirst_flickering2','mvt_type',
                      'sequence_at_least_x_repetition'], inplace=True, axis=1)
        #zone: initial record name. the consecutives equal zones has been removed (but not that all variables were computed with them
        # for exemple flickering type 1 would have less eveent if these consecutives equal zones would have been removed beforehand)
        #Zone_without_flickering: name of the zone defined by the flickering type 1 rule, if empty it means that the row should be
        #removed (i.e. belongs to an interzone and is not the first record of this interzone event)
        #flickering2_name: flickering2 type if its the first record of such a sequence
        #is_impossible_mvt: True if the next record zone is actually not a possible zone (again, note that a lot of those situation 
        #might be caused by two same timestamp record with two different zones. We should decide if we can use the order of the
        #initial record file or not for this)
        #Now we need to define rule based basically on these columns so that we could remove the unwanted records 
        
        ##### finding consecutives flickering2 events
        df_info['flickering2_name'] = df_info['flickering2_name'].fillna('')
        li_df = []
        for i, df_hen in tqdm.tqdm(df_info.groupby(['HenID'])):
            df_hen['previous_zone_flickering2'] = [None]+df_hen['zone_flickering2'].tolist()[0:-1]
            df_hen['is_of_interest'] = df_hen.apply(lambda x: (x['flickering2_name']!='') & \
                                                    (x['previous_zone_flickering2']=='flickering2'), axis=1)
            df_hen['interesting_sequ'] = df_hen.apply(lambda x: [i for i in df_hen[df_hen['Timestamp']<x['Timestamp']]['flickering2_name'].tolist() if i!=''][-1] if x['is_of_interest'] else None, axis=1)
            li_df.append(df_hen)
        #put again in one dataframe
        df_info = pd.concat(li_df) 
        
        ##### save
        df_info.to_csv(os.path.join(path_extracted_data, id_run+'_records_with_some_cleaned_info_'+str(nbr_block_repetition)+'.csv'),
                       index=False, sep=';')
        t2 = time.clock()
        print ("Running time for saving and aggregating info to base rule on is: %.2f mn" %((t2-t1)/60))
    
    #print('Note that the order of removing impact the error associated to each event')
    #ceiz
    #x0 = df.shape[0]
    #df = df[~df['correction_is_consecutive_equal_initial_zone']] 
    #print_color((('We have ','black'),(df.shape[0],'green'),(' records (','black'),(x0-df.shape[0],'red'),
    #         (' removed due to consecutives equal initial zones)','black')))

    #flickeringtype2 
    #x0 = df.shape[0]
    #df = df[~df['TODOOOO keep only first among all the existing one?']] 
    #print_color((('We have ','black'),(df.shape[0],'green'),(' records (','black'),(x0-df.shape[0],'red'),
    #         (' removed due to consecutives equal initial zones)','black')))

    #flickeringtype1
    #x0 = df.shape[0]
    #df = df[~df['test_ToRemove_flickering1']] 
    #print_color((('We have ','black'),(df.shape[0],'green'),(' records (','black'),(x0-df.shape[0],'red'),
    #         (' removed due to consecutives equal initial zones)','black')))

    #if save:
    #    df.filter(['HenID','Timestamp', ???]).to_csv(os.path.join(path_extracted_data,
    #id_run+'_clean_record_'+str(nbr_block_repetition)+'.csv'),index=False, sep=';')

    #runing time info and return final cleaned df
    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))
    
    return df
    


##########################################################################################################################################
############################################################# time series ################################################################
##########################################################################################################################################


def time_series_henColumn_tsRow(df, config, col_ts='Zone' , name_='', ts_with_all_hen_value=True, save=True, 
                                hen_time_series=False):
    
    '''one time series with each column being one hen. becasue then onpeing one file we have all. also, no need to go column by column to change day. '''
    
    #initialize parameter
    path_extracted_data = config.path_extracted_data
    id_run = config.id_run
    nbr_sec = config.nbr_sec
    
    #create a director if not existing
    if not os.path.exists(path_extracted_data):
        os.makedirs(path_extracted_data)
        
    #verify columns name if not done before: TODO
    
    print('in this time series there is %d hens'%len(df['HenID'].unique()))
    
    #sort by timestamp and do pivot
    df = df.sort_values(['Timestamp'], ascending=True)
    x0 = df.shape[0]
    #No need now that no duplicate timestamp per hen:
    #df = df.groupby(['Timestamp','HenID'])[col_ts].agg(lambda x: list(x)[0]).reset_index()  
    df_hens = df.pivot(index='Timestamp', columns='HenID', values=col_ts)

    #fill "None" values with the last non-empty value (by propagating last valHenID observation forward to next valHenID)
    #In order to fill in between timestamp. Note that the first one will stay None
    df_hens = df_hens.fillna(method='ffill')

    #add missing dates
    mi = min(df['Timestamp'].tolist())
    ma = max(df['Timestamp'].tolist())
    print('The initial starting date of the time series is: %s, and the ending date will be: %s'%(str(mi), str(ma)))
    
    #add dates until minuit of the last day
    ma = dt.datetime(ma.year,ma.month,ma.day,23,59,59)
    print('and after ending the last day at midnight : %s, and the ending date will be: %s'%(str(mi), str(ma)))
    Daterange = pd.date_range(start=mi, end=ma, freq='S') 

    #take only the needed seconds (nbr_sec)
    Daterange = [Daterange[i] for i in range(len(Daterange)) if i%nbr_sec==0]
    #Daterange[0:10]
    
    #add missing seconds and fillnan with last non-nan values by propagating last valHenID observation (even if its an observation
    #that will be removed) forward to next valHenID
    df_hens = df_hens.reindex(Daterange, method='ffill').reset_index()
    #df_hens.tail(20)
    
    #remove timestamp without all hen, if requested
    df_hens['nbr_nan'] = df_hens.isnull().sum(axis=1)
    #plt.plot(df_hens['nbr_nan']);
    print(df_hens['nbr_nan'].unique())
    if ts_with_all_hen_value:
        #df_hens[10729:]['nbr_nan'].unique() #only 0 after the first one
        df_hens = df_hens[df_hens['nbr_nan']==0]
        df_hens.drop(['nbr_nan'], inplace=True, axis=1)
        print('as we want the time series to start at the same time, we remove the dates without info on each hen, making us start on ',
              df_hens.iloc[0]['Timestamp'])
    
    #remove dates with health care
    df_hens['day'] = df_hens['Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day))
    df_hens['hour'] = df_hens['Timestamp'].map(lambda x: x.hour if pd.isnull(x)!=np.nan else np.nan)
    li_date2remove = config.li_date2remove
    if len(li_date2remove)!=0:
        #li_date2remove = [dt.datetime.strptime(x, '%Y-%m-%d') for x in li_date2remove]
        df_hens['date_toberemoved'] = df_hens['day'].map(lambda x: x in li_date2remove)
        #df['date_toberemoved'].value_counts()
        x0 = df_hens.shape[0]
        df_hens = df_hens[~df_hens['date_toberemoved']]
        print_color((('By removing the health-assessement days we passed from %d to %d timestamp (losing '%(x0,
                    df_hens.shape[0]),'black'), (x0-df_hens.shape[0],'red'),(' timestamp)','black')))

    if save:
        df_hens.to_csv(os.path.join(path_extracted_data,id_run+'_TimeSeries_'+str(name_)+'.csv'), sep=';', index=False)

    #one time serie per hen
    if hen_time_series:
        #remove the day keep only the timestamp
        df_hens['Timestamp_value'] = df_hens['Timestamp'].map(lambda x: dt.timedelta(hours=x.hour, 
                                                                                       minutes=x.minute, 
                                                                                       seconds=x.second))        
        li_hen = [h for h in df_hens.columns if h.startswith('hen_')]
        for h in li_hen:
            #select the column associated to the hen
            df_per_hen = df_hens[[h,'Timestamp_value','day']].copy()
            df_per_hen[h] = df_per_hen[h].map(lambda x: int(x[5:]) if x!=None else np.nan)
            #pivot, to put the day in column intead of having one row for each timestamp_value per day
            df_per_hen = df_per_hen.pivot(index='Timestamp_value', columns='day', values=h)
            df_per_hen.reset_index(drop=False, inplace=True)
            df_per_hen.to_csv(os.path.join(config.path_extracted_data, id_run+'_TimeSeries_'+str(name_)+'_'+h+'.csv'), sep=';')

    return(df_hens)


##########################################################################################################################################
######################################################### variable computation ###########################################################
##########################################################################################################################################

#computing chi2-distance
def chi2_distance(l1,l2):
    '''sompute the following distance: d(x,y) = sum( (xi-yi)^2 / (xi+yi) ) / 2'''
    if len(l1)!=len(l2):
        print('your two vectors must have same length')
        sys.exit()
    if (sum(l1)<0.99) | (sum(l1)>1.0001) | (sum(l2)<0.99) & (sum(l2)>1.0001):
        print('your two vectors must be normalized (sumed to one)')
        sys.exit()
    d = sum([(l1[i]-l2[i])**2 / ((l1[i]+l2[i])+0.000000000001) for i in range(len(l1))])/2
    return(d)
#m1 = [1,1,2,2,3,3,3,3]
#m2 = [1,3,3,3,3,3,2,2]
#l1 = [Counter(m1)[i]/len(m1) for i in range(1,nbr_topics+1)]
#l2 = [Counter(m2)[i]/len(m2) for i in range(1,nbr_topics+1)]
#chi2_distance(l1,l2)

def sequence_without_cez(li):
    '''function to compute number of time a hen went into a zone'''
    v = [i[0] for i in list(itertools.groupby(li, lambda x: x))]
    return v
#small test
#t = [1,1,1,1,2,2,2,3,3,3,4,4,2,2,2,2]
#sequence_without_cez(t) #--> [1, 2, 3, 4, 2]

def nbr_bouts_per_zone(li):
    '''function to compute number of time a hen went into a zone'''
    v = [i[0] for i in list(itertools.groupby(li, lambda x: x))]
    return dict(Counter(v))
#small test
#t = [1,1,1,1,2,2,2,3,3,3,4,4,2,2,2,2]
#nbr_bouts_per_zone(t) #--> {1: 1, 2: 2, 3: 1, 4: 1}    

def nbr_transition(li):
    '''function to compute number of transition from list of consecutives zone name'''
    return max(len(list(itertools.groupby(li, lambda x: x)))-1,0) #max in case of empty list
#small test
#t = [1,1,1,1,2,2,2,3,3,3,4,4,2,2,2,2]
#nbr_transition(t) #4

def max_duration_zones(li):
    '''function to find zone(s) where the hen staid longer'''
    v = [(x[0], len(list(x[1]))) for x in itertools.groupby(li)]
    v = sorted(v, key = lambda i: i[1])
    m = max([i[1] for i in v])
    v = [i[0] for i in v if i[1]==m]   
    if len(v)==1:
        return(v[0])
    else:
        print('several max-duration-zone')
        return(v)

############statistics on duration
#where for li = [1,1,1,1,2,2,2,3,3,3,4,4,2,2,2,2] we get if nbr_sec=3: v= [12, 9, 9, 6, 12] which we then aggreagate (min, max, avg,...)
def list_of_durations(li, nbr_sec):
    return [len(list(x[1]))*nbr_sec for x in itertools.groupby(li)]

def max_duration(li, nbr_sec):
    '''function to compute the maximum duration in any zone'''
    v = [len(list(x[1]))*nbr_sec for x in itertools.groupby(li)]
    return max(v)

def var_duration(li, nbr_sec):
    '''function to compute the variation duration in any zone'''
    #list of durations
    v = [len(list(x[1]))*nbr_sec for x in itertools.groupby(li)]
    return np.var(v) 

def min_duration(li, nbr_sec):
    '''function to compute the min duration in any zone'''
    #list of durations
    v = [len(list(x[1]))*nbr_sec for x in itertools.groupby(li)]
    return min(v) 

def median_duration(li, nbr_sec):
    '''function to compute the median duration in any zone'''
    v = [len(list(x[1]))*nbr_sec for x in itertools.groupby(li)]
    return np.median(v)

def average_duration(li, nbr_sec):
    '''function to compute the average duration in any zone'''
    v = [len(list(x[1]))*nbr_sec for x in itertools.groupby(li)]
    return np.mean(v)
############

def dico_zone_duration(li, nbr_sec):
    '''function to find a list of duration per zone'''
    v = [(x[0], len(list(x[1]))) for x in itertools.groupby(li)]
    v = sorted(v, key = lambda i: i[1])
    d = {}
    for i,j in v:
        if i not in d:
            d[i] = []
        d[i].append(j*nbr_sec)
    return d    
#small test
#t = [1,1,1,1,2,2,2,3,3,3,4,4,2,2,2,2,]
#max_duration(t), min_duration(t), median_duration(t), max_duration_zones(t), average_duration(t), dico_zone_duration(t)
#--> several max-duration-zone
#(12, 6, 9.0, [1, 2], 9.600000000000001, {1: [12], 2: [9, 12], 3: [9], 4: [6]})


#not linked to the order, only the distribution (proba of a certain zone)
def DistributionEntropy(labels):
    '''compute the distribution entropy'''
    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts)
#small test
#print(DistributionEntropy([1,1,1,1,1,1,1,1,0,2,3,4,5,6,7,8,9,9,9,9]), 
#      DistributionEntropy([0,2,3,4,5,6,7,8,9,9,9,9,1,1,1,1,1,1,1,1]),
#      DistributionEntropy([1,1,1,1,1,1,1,1]), 
#      DistributionEntropy([1,0,1,1,1,1,1,1,0]), 
#      DistributionEntropy([0,1,1,1,1,1,1,0,1]), 
#      DistributionEntropy([1,1,0,0,1,1,1,1,1]),
#      DistributionEntropy(['1','1','8','9','9'])) #categorical values oke
#1.8866967846580784 1.8866967846580784 0.0 0.5297061990576545 0.5297061990576545 0.5297061990576545

def stats_list(li_en_shift):
    dico = {}
    index_max = li_en_shift.index(max(li_en_shift))
    index_min = li_en_shift.index(min(li_en_shift))
    dico['is_max_first'] = index_max<index_min
    dico['abs_min_max'] = abs(min(li_en_shift)-max(li_en_shift))
    dico['min'] = min(li_en_shift)
    dico['max'] = max(li_en_shift)
    dico['variance'] = np.var(li_en_shift)
    dico['median'] = np.median(li_en_shift)
    dico['mean'] = np.mean(li_en_shift)
    return dico

def stats_on_shifted_SampEnt(li, NbrData, ValueDelta, EntropyTimeComputation):
    NbrValues = math.ceil(NbrData/ValueDelta)
    li_en_shift = [sample_entropy(li[(i-NbrValues):i], order=2, 
                                        metric='chebyshev') for i in range(NbrValues, len(li), EntropyTimeComputation)]
    #print(li_en_shift)
    #print(range(NbrValues, len(li), EntropyTimeComputation))
    #print([li[(i-NbrValues):i] for i in range(NbrValues, len(li), EntropyTimeComputation)])
    return stats_list(li_en_shift)
#small exemple
#population = [1, 2, 3, 4, 5, 6] ; weights = [0.1, 0.05, 0.05, 0.2, 0.4, 0.2]
#li = random.choices(population, weights, k=100)
#print(li)
#stats_on_shifted_SampEnt(li=li, NbrData=20, ValueDelta=2, EntropyTimeComputation=10)

def stats_on_running_SampEnt(li, NbrData, ValueDelta, EntropyTimeComputation):
    NbrValues = math.ceil(NbrData/ValueDelta)
    li_en_shift = [sample_entropy(li[0:i], order=2, metric='chebyshev') for i in range(NbrValues, len(li), EntropyTimeComputation)]
    return stats_list(li_en_shift)

def stats_on_shifted_DistEnt(li, NbrData, ValueDelta, EntropyTimeComputation):
    NbrValues = math.ceil(NbrData/ValueDelta)
    li_en_shift = [DistributionEntropy(li[(i-NbrValues):i]) for i in range(NbrValues, len(li), EntropyTimeComputation)]
    return stats_list(li_en_shift)

def stats_on_running_DistEnt(li, NbrData, ValueDelta, EntropyTimeComputation):
    NbrValues = math.ceil(NbrData/ValueDelta)
    li_en_shift = [DistributionEntropy(li[0:i]) for i in range(NbrValues, len(li), EntropyTimeComputation)]
    return stats_list(li_en_shift)

def avg_diff_linbr(li):
    '''compute the avergae difference between any combination of two numbers in a list of numbers'''
    r = []
    for i,x in enumerate(li[:-1]):
        for y in li[i+1:]:
            r.append(abs(x-y))
    return np.mean(r)
#small e.g.
#avg_diff_linbr([7,8,9]) #1,3





def ZoneVariable(df_ts, config, save=True, red_dot_for_each_hen=True, nbr_bird_per_square_meter=False):

    '''From a time series (one column per hen named by 'hen_'), compute a Heatmap of number of birds in each zone at each 
    timestamp we are taking one value per minute (the first one), and we are not considering the rest
    red_dot_for_each_hen: if True, then we will plot where each bird is with a red dot in order to understand his synchronicity with other birds and if he likes crowd and when. It can then help extract some variables of interest
    nbr_bird_per_square_meter: If True, the nbr of birds will be divided by the umber of square meter associated to that zone'''
    
    #start recording the time it last
    START_TIME = time.clock()
    
    #initialise variable
    path_extracted_data = config.path_extracted_data
    dico_zone_order = config.dico_zone_order
    id_run = config.id_run
    dico_zone_plot_name = config.dico_zone_plot_name
    dico_zone_meter2 = config.dico_zone_meter2

    df_ts['minute'] = df_ts['Timestamp'].map(lambda x: x.minute)
    li_hen = [i for i in df_ts.columns if i.startswith('hen_')]
    li_zones = list(df_ts[li_hen].stack().unique())
    plot_type = 'number of birds'
    if nbr_bird_per_square_meter:
        plot_type = plot_type+' per m2'
    #sort the yaxis for the naming
    s = sorted(dico_zone_order.items(), key=operator.itemgetter(1))
    s = {x[1]:dico_zone_plot_name[x[0]] for x in s}

    #create path where to save if not existing yet
    path_ = os.path.join(path_extracted_data,'visual','Nbr_bird_In_Zone')
    #create a director if not existing
    if not os.path.exists(path_):
        os.makedirs(path_)

    #for each day draw a heatmap
    for day in tqdm.tqdm(df_ts['day'].unique()):
        df_ = df_ts[df_ts['day']==day].sort_values(['Timestamp'])
        #xaxis might be different over the days, if not complete days, so we will take the appropriate timestamp
        #take only the smallest timestamp per minute
        Xaxis = df_.groupby(['hour','minute'])['Timestamp'].agg(lambda x: min(list(x))).reset_index()['Timestamp'].tolist()       
        M = np.zeros(shape=(max(dico_zone_order.values())+1, len(Xaxis))) #+1 car starts from 0
        for i,ts in enumerate(Xaxis):
            #list of all zones happening on a particular timestamp that day
            li = list(df_[df_['Timestamp']==ts][li_hen].values[0])
            c = Counter(li)
            #print(sum(list(c.values()))) 
            for zone_, order in dico_zone_order.items():
                if zone_ in c:
                    M[order][i] = c[zone_]
                    if nbr_bird_per_square_meter:
                        M[order][i] = M[order][i] / dico_zone_meter2[zone_]

        #plot and save
        #plt.figure()
        plt.clf() # clears the entire current figure instead of plt.figure() which will create a new one, and hence keeping all figures
        #in memory
        #fig, ax = plt.subplots(figsize=(10,8))         #figsize in inches
        sns.set(font_scale=0.6) 
        ax = sns.heatmap(M, cmap="YlGnBu", yticklabels=[s.get(j,' ') for j in range(M.shape[0])],
                   xticklabels=[':'.join(str(Xaxis[i]).split(' ')[1].split(':')[0:2]) if i%30==0 else '' for i in range(len(Xaxis))])  
        ax.invert_yaxis()
        plt.title(str(day).split('T')[0] +'      '+plot_type)
        if save:
            plt.savefig(os.path.join(path_,id_run+'_'+plot_type+'_'+str(day).split('T')[0]+'.png'), format='png', dpi=300)
        #plt.show()
        plt.close()
        
        #add a red point for each hen and save the hen plot
        dico_zone_order_ = dico_zone_order.copy()
        dico_zone_order_['nan'] = -0.5
        if red_dot_for_each_hen:
            for hen_ in li_hen:
                #plot the whole heatmap again 
                path_plt = os.path.join(path_, id_run+'_'+plot_type+'_'+hen_+'_'+str(day).split('T')[0]+'.png')
                p = glob.glob(path_plt)
                if len(p)==1:
                    continue
             
                plt.clf()
                ax = sns.heatmap(M, cmap="YlGnBu", yticklabels=[s.get(j,' ') for j in range(M.shape[0])],
                           xticklabels=[':'.join(str(Xaxis[i]).split(' ')[1].split(':')[0:2]) if \
                                        i%30==0 else '' for i in range(len(Xaxis))])  
                ax.invert_yaxis()
                plt.title(str(day).split('T')[0]+'      '+plot_type+' and '+hen_ +' (red)')
                #add info of the hen
                li_zone_hen = df_[df_['Timestamp'].isin(Xaxis)][hen_].tolist()
                li_zone_hen = [dico_zone_order_[str(x)]+0.5 for x in li_zone_hen] #0.5 to show it in the middle of the heatmap bar
                ax.scatter(range(len(Xaxis)), li_zone_hen, marker='d', s=1, color='red') #s = size
                if save:
                    plt.savefig(path_plt, format='png', dpi=300, bbox_inches='tight') 
                #plt.show()    
                plt.close()
                
    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))  
    
    
def ZoneVariable_old(df_ts, config, save=True):
    
    '''From a time series (one column per hen named by 'hen_'), compute a Heatmap of number of birds in each zone at each timestamp
    we are taking one value per minute (the first one), and we are not considering the rest'''
        
    #start recording the time it last
    START_TIME = time.clock()
    
    #initialise variable
    path_extracted_data = config.path_extracted_data
    dico_zone_order = config.dico_zone_order
    id_run = config.id_run
    
    df_ts['minute'] = df_ts['Timestamp'].map(lambda x: x.minute)
    li_hen = [i for i in df_ts.columns if i.startswith('hen_')]
    li_zones = list(df_ts[li_hen].stack().unique())

    #sort the yaxis
    s = sorted(dico_zone_order.items(), key=operator.itemgetter(1))

    #create path where to save if not existing yet
    path_ = os.path.join(path_extracted_data,'visual','Nbr_bird_In_Zone')
    #create a director if not existing
    if not os.path.exists(path_):
        os.makedirs(path_)

    #for each day draw a heatmap
    for day in tqdm.tqdm(df_ts['day'].unique()):
        df_ = df_ts[df_ts['day']==day].sort_values(['Timestamp'])
        #xaxis might be different over the days, if not complete days, so we will take the appropriate timestamp
        #take only the smallest timestamp per minute
        Xaxis = df_.groupby(['hour','minute'])['Timestamp'].agg(lambda x: min(list(x))).reset_index()['Timestamp'].tolist()       
        M = np.zeros(shape=(max(dico_zone_order.values())+1, len(Xaxis)))
        for i,ts in enumerate(Xaxis):
            #list of all zones happening on a particular timestamp that day
            li = list(df_[df_['Timestamp']==ts][li_hen].values[0])
            c = Counter(li)
            #print(sum(list(c.values()))) #112
            for zone_, order in dico_zone_order.items():
                if zone_ in c:
                    M[order][i] = c[zone_] 
        #plot and save
        plt.figure()
        #fig, ax = plt.subplots(figsize=(10,8))         #figsize in inches
        sns.set(font_scale=0.6) 
        ax = sns.heatmap(M, cmap="YlGnBu", yticklabels=[i[0] for i in s],
                   xticklabels=[':'.join(str(Xaxis[i]).split(' ')[1].split(':')[0:2]) if i%30==0 else '' for i in range(len(Xaxis))])  
        ax.invert_yaxis() 
        plt.title(str(day).split('T')[0])
        if save:
            plt.savefig(os.path.join(path_,id_run+'_nbr_birds_in_each_zone_nointerzone_'+str(day).split('T')[0]+'.png'), dpi=300,
                    format='png',bbox_inches='tight')            
        plt.close()
        
    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))  
    
    
    
def OneTimeSeriesPlot(df_ts, config, timestamp_name='New_Timestamp', value='most_frequent_zone', save=True, 
                      last_folder_name='', name_=''):
    
    '''For one hen timeseries plot the time series over the whole period
    df_ts should have at least those columns: timestamp_name, hen_*, value, day'''
    
    #initialise variable
    path_extracted_data = config.path_extracted_data
    dico_zone_order = config.dico_zone_order
    id_run = config.id_run

    li_day = df_ts['day'].unique()
    hen_name = [x for x in df_ts.columns if x.startswith('hen_')][0]
    if name_=='':
        name_ = hen_name
        
    #sort the yaxis
    s = sorted(dico_zone_order.items(), key=operator.itemgetter(1))
    li_zone_in_order = [x[0] for x in s]

    #create path where to save if not existing yet
    if last_folder_name=='':
        path_ = os.path.join(path_extracted_data,'visual','TimeSeriesPlot')
    else:
        path_ = os.path.join(path_extracted_data,'visual','TimeSeriesPlot',last_folder_name)
    #create a director if not existing
    if not os.path.exists(path_):
        os.makedirs(path_)

    #sort by dates to make sure
    df_ts = df_ts.sort_values([timestamp_name])
    
    fig, ax = plt.subplots(figsize=(25,3))
    #plot
    df_ts['zone_ts'] = df_ts[value].map(lambda x: int(dico_zone_order[x])).tolist()
    plt.plot(df_ts[timestamp_name].tolist(), df_ts['zone_ts'])
    for day in li_day : 
        plt.axvline(x=day, ymin=min(dico_zone_order.values()), ymax=max(dico_zone_order.values()), linewidth=1, color='r')
    plt.title(name_) 
    #match yaxis with the true zone names
    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_yticklabels()]
    #convert to flaot as sometimes its float (when less or more zones)
    labels = [str(float(x.replace('−','-'))) for x in labels]
    for name,z in dico_zone_order.items():
        if str(float(z)) in labels:
            labels[labels.index(str(float(z)))] = name
    labels = [i if i in dico_zone_order.keys() else '' for i in labels]
    ax.set_yticklabels(labels)
    #rotate xaxis
    labels = [item.get_text() for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=0)
    #save
    if save:
        plt.savefig(os.path.join(path_,id_run+'_ts_'+name_+'.png'), format='png')
    plt.show();
    plt.close()
    
    
def TimeSeriesPlot(df_ts, config, save=True, last_folder_name='', name_=''):
    
    '''For a csv with one column=one time series, plot all the time series'''
    
    #start recording the time it last
    START_TIME = time.clock()

    #initialise variable
    path_extracted_data = config.path_extracted_data
    dico_zone_order = config.dico_zone_order
    id_run = config.id_run

    li_hen = [i for i in df_ts.columns if i.startswith('hen_')]
    li_zones = list(df_ts[li_hen].stack().unique())
    li_day = df_ts['day'].unique()

    #sort the yaxis
    s = sorted(dico_zone_order.items(), key=operator.itemgetter(1))
    li_zone_in_order = [x[0] for x in s]
    
    #create path where to save if not existing yet
    if last_folder_name=='':
        path_ = os.path.join(path_extracted_data,'visual','TimeSeriesPlot')
    else:
        path_ = os.path.join(path_extracted_data,'visual','TimeSeriesPlot',last_folder_name)
    #create a director if not existing
    if not os.path.exists(path_):
        os.makedirs(path_)

    #create path where to save if not existing yet
    path_ = os.path.join(path_extracted_data,'visual','TimeSeriesPlot')
    #create a director if not existing
    if not os.path.exists(path_):
        os.makedirs(path_)

    #for each hen draw a timeseries
    for hens in tqdm.tqdm(li_hen):
        fig, ax = plt.subplots(figsize=(25,3))
        df_plt = df_ts[~df_ts[hens].isnull()].sort_values(['Timestamp']).copy()
        zone_ts = df_plt[hens].map(lambda x: int(dico_zone_order[x])).tolist()
        plt.plot(df_plt['Timestamp'].tolist(), zone_ts)
        for day in li_day : 
            plt.axvline(x=day, ymin=min(dico_zone_order.values()), ymax=max(dico_zone_order.values()), linewidth=1, color='r')
        plt.title(hens)
        #rotate xaxis
        fig.canvas.draw()
        labels = [item.get_text() for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=0)
        if save:
            plt.savefig(os.path.join(path_,id_run+'_TimeSeries_'+name_+'_'+hens+'.png'), dpi=300, format='png',bbox_inches='tight')       
        plt.show();
        plt.close()
        
    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))  

    
def ts_compare_session(HenID, SessionID1, SessionID2, config, save=True, title='', last_folder_name='session_comparison'):
    
    #initialise variable
    path_extracted_data = config.path_extracted_data
    dico_zone_order = config.dico_zone_order
    id_run = config.id_run
    
    #sort the yaxis
    s = sorted(dico_zone_order.items(), key=operator.itemgetter(1))
    li_zone_in_order = [x[0] for x in s]
    
    #sort the yaxis
    s = sorted(dico_zone_order.items(), key=operator.itemgetter(1))
    li_zone_in_order = [x[0] for x in s]    
    
    #create path where to save if not existing yet
    if last_folder_name=='':
        path_ = os.path.join(path_extracted_data,'visual','TimeSeriesPlot')
    else:
        path_ = os.path.join(path_extracted_data,'visual','TimeSeriesPlot',last_folder_name)
    #create a director if not existing
    if not os.path.exists(path_):
        os.makedirs(path_)
        
    #open timeseries of each session
    df_ts1 = pd.read_csv(os.path.join(path_extracted_data, id_run+'_TimeSeries_'+SessionID1+'.csv'),
                    sep=';', parse_dates=['Timestamp', 'day']) 
    df_ts2 = pd.read_csv(os.path.join(path_extracted_data, id_run+'_TimeSeries_'+SessionID2+'.csv'),
                        sep=';', parse_dates=['Timestamp', 'day']) 
    df_ts1 = df_ts1.sort_values('Timestamp', ascending=True)
    df_ts2 = df_ts2.sort_values('Timestamp', ascending=True)
    #make the ts start from begining
    mi1 =  min(df_ts1['Timestamp'].tolist())
    df_ts1['Timestamp'] = df_ts1['Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day,0,0,0) if dt.datetime(x.year,x.month,x.day,x.hour,x.minute,x.second)==dt.datetime(mi1.year,mi1.month,mi1.day,1,30,0) else x)
    mi2 =  min(df_ts2['Timestamp'].tolist())
    df_ts2['Timestamp'] = df_ts2['Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day,0,0,0) if dt.datetime(x.year,x.month,x.day,x.hour,x.minute,x.second)==dt.datetime(mi2.year,mi2.month,mi2.day,1,30,0) else x)
    df_ts1['zone_ts'] = df_ts1[HenID].map(lambda x: int(dico_zone_order[x])).tolist()
    df_ts2['zone_ts'] = df_ts2[HenID].map(lambda x: int(dico_zone_order[x])).tolist()
    li_day1 = sorted(df_ts1['day'].unique()) #sort du plus petit au plus grand
    li_day2 = sorted(df_ts2['day'].unique())     
    
    ################### plot
    fig, axs = plt.subplots(max(len(li_day1),len(li_day2)), 2, constrained_layout=True, figsize=(20,8))
    fig.suptitle(title) 
    
    # first session
    for i,d in enumerate(li_day1):
        #df_plt = df_ts1[(~df_ts1[HenID].isnull())&(df_ts1['day']==d)].copy()
        df_plt = df_ts1[df_ts1['day']==d].copy()
        axs[i,0].plot(df_plt['Timestamp'].tolist(), df_plt['zone_ts'].tolist())
        axs[i,0].set_title(str(d).split('T')[0], size=8)
            
    # second session
    for i,d in enumerate(li_day2):
        #df_plt = df_ts2[(~df_ts2[HenID].isnull())&(df_ts2['day']==d)].copy()
        df_plt = df_ts2[df_ts2['day']==d].copy()
        axs[i,1].plot(df_plt['Timestamp'].tolist(), df_plt['zone_ts'].tolist())
        axs[i,1].set_title(str(d).split('T')[0], size=8)

    #save
    if save:
        plt.savefig(os.path.join(path_,id_run+'_ts_'+HenID+'_'+SessionID1+'_'+SessionID2+'.png'), dpi=300, format='png')
        plt.show();
    plt.close()
    
#small exemple
#ts_compare_session('hen_105', '3B', '5B', config, last_folder_name='session_comparison')


def entropy_compare_session(h, SessionID1, SessionID2, config, title, last_folder_name='', value_delta=30, time_delta=60*5):
    '''compute running sample entropy'''
    
    #initialise variable
    path_extracted_data = config.path_extracted_data
    dico_zone_order = config.dico_zone_order
    id_run = config.id_run
        
    path_entropy = os.path.join(path_extracted_data,'visual','entropy',last_folder_name)
    #create a director if not existing
    if not os.path.exists(path_entropy):
        os.makedirs(path_entropy)
        
    #open timeseries of the two sessions
    df_ts1 = pd.read_csv(os.path.join(path_extracted_data, id_run+'_TimeSeries_'+SessionID1+'.csv'),
                         sep=';', parse_dates=['Timestamp','day']) 
    df_ts2 = pd.read_csv(os.path.join(path_extracted_data, id_run+'_TimeSeries_'+SessionID2+'.csv'),
                        sep=';', parse_dates=['Timestamp','day']) 
    df_ts1 = df_ts1.sort_values('Timestamp', ascending=True)
    df_ts2 = df_ts2.sort_values('Timestamp', ascending=True)
    
    #compute var in order to set axis of plot correctly
    li_zones1 = list(df_ts1[h].unique())
    li_zones2 = list(df_ts2[h].unique())
    #without having to follow a particular order
    fig, axs = plt.subplots(max(len(li_zones1),len(li_zones2))+2, 2, constrained_layout=True, figsize=(20,8))
    fig.suptitle(title+'\n \n ', size=16) 
    
    for df_ts,c,li_zones in [(df_ts1,0,li_zones1), (df_ts2,1,li_zones2)]:
                
        #nan at the begining should not influence entropy (middle cant exist by construction) so lets remove it to have
        #timestamp according to this  
        df_ts = df_ts[~df_ts[h].isnull()]
        df_ts[h] = df_ts[h].map(dico_zone_order).astype(int)
        
        #compute all-zones SampEtn and DistributionEnt
        li_ts = [x for i,x in enumerate(df_ts['Timestamp'].tolist()) if i%time_delta==0]
        li_ = [x for i,x in enumerate(df_ts[h].tolist()) if i%time_delta==0]
        range_ = range(value_delta, len(li_))
        li_ts_xaxis = [li_ts[i] for i in range_]
        #DistribEnt
        li_en = [DistributionEntropy(li_[0:i]) for i in range_]
        li_en_running = [DistributionEntropy(li_[(i-value_delta):i]) for i in range_]
        axs[0,c].set_ylim((0,1.5))
        axs[0,c].plot(li_ts_xaxis, li_en_running)
        axs[0,c].plot(li_ts_xaxis, li_en)
        axs[0,c].set_title('Distribution Entropy All Zones', size=11)       
        #SampEnt
        li_en = [SampEnt(li_[0:i]) for i in range_]
        li_en_running = [SampEnt(li_[(i-value_delta):i]) for i in range_]
        axs[1,c].set_ylim((0,1))
        axs[1,c].plot(li_ts_xaxis, li_en_running)
        axs[1,c].plot(li_ts_xaxis, li_en)
        axs[1,c].set_title('SampEnt All Zones', size=11) 
        
        #compute per zone entropy
        for k,zone_ in enumerate(li_zones):
            df_ts_zone = df_ts.copy()
            df_ts_zone[h] = df_ts_zone[h].map(lambda x: x==zone_)
            li_ts = [x for i,x in enumerate(df_ts_zone['Timestamp'].tolist()) if i%time_delta==0]
            li_ = [x for i,x in enumerate(df_ts_zone[h].tolist()) if i%time_delta==0]
            range_ = range(value_delta, len(li_))
            li_en = [SampEnt(li_[0:i]) for i in range_]
            li_en_running = [SampEnt(li_[i-value_delta:i]) for i in range_]
            li_ts_xaxis = [li_ts[i] for i in range_]
            axs[k+2,c].set_ylim((0,1))
            axs[k+2,c].plot(li_ts_xaxis, li_en_running)
            axs[k+2,c].plot(li_ts_xaxis, li_en)
            axs[k+2,c].set_title('SampEnt '+zone_, size=11)
            
    #save plot
    plt.savefig(os.path.join(path_entropy, id_run+'_entropy_'+h+'_'+SessionID1+'_'+SessionID2+'.png'), 
                dpi=300, format='png')
    plt.show()   
    plt.close()
    
    
def is_day(x,dico_night_hour):
    '''from a timestamp value x, and the dico_nighthour parameter, it will output true if its during the day, false otherwise'''
    if max(dico_night_hour.keys())<dt.datetime(x.year,x.month,x.day,0,0,0):
        print('ERROR: your \"dico_night_hour\" parameter does not include information for the date: %s'%str(x))
        sys.exit()
    else:
        #take info (i.e. values) of the dico_night_hour key that represent the smallest date among all the date>=x:
        m = min([d for d in dico_night_hour.keys() if d>=dt.datetime(x.year,x.month,x.day,0,0,0)])
        #is the timestamp smaller than the latest day hour and bigger than the first day hour? 
        #Attention-limitation: midnight should not be included in the day hour
        return((dt.datetime(1,1,1,x.hour,x.minute,0)>=dt.datetime(1,1,1,dico_night_hour[m]['start_day_h'],dico_night_hour[m]['start_day_m'],0)) & \
        (dt.datetime(1,1,1,x.hour,x.minute,0)<dt.datetime(1,1,1,dico_night_hour[m]['end_day_h'],dico_night_hour[m]['end_day_m'],0)))

def correct_key(x,dico_night_hour):
    '''from a timsetamp and the dico_night_hour parameter, it will output the key of the dico_night_hour that is associated to the
    timestamp x'''
    return min([d for d in dico_night_hour.keys() if d>=dt.datetime(x.year,x.month,x.day,0,0,0)])


def day_duration(t1, t2, dico_night_hour):
    '''from two timestamp and the dico_night_hour parameter it will ouput the duration inbetween unless it may be biaised by 
    sleeping time, in which case it will return -1'''
    #Put -1 if at least one of the two timestamp happens during the night or if none happend during the night but the second 
    #timestamp is later than the last possible daily-timestamp of the day of the first timestamp
    if (is_day(t2,dico_night_hour) + is_day(t1,dico_night_hour))!=2:
        return -1
    elif t2>dt.datetime(t1.year, t1.month, t1.day, 
                        dico_night_hour[correct_key(t1,dico_night_hour)]['end_day_h'],
                        dico_night_hour[correct_key(t1,dico_night_hour)]['end_day_m'],0):
        return -1
    else:
        return (t2-t1).seconds
#some verification   
#day_duration(dt.datetime(2019,8,9,11,1,1),dt.datetime(2019,8,9,11,1,1),dico_night_hour) #0
#day_duration(dt.datetime(2019,8,9,11,1,1),dt.datetime(2019,8,9,11,1,3),dico_night_hour) #2
#day_duration(dt.datetime(2019,8,9,11,1,1),dt.datetime(2019,8,10,11,1,1),dico_night_hour) #-1
#day_duration(dt.datetime(2019,8,9,16,1,1),dt.datetime(2019,8,9,23,1,1),dico_night_hour) #-1
#day_duration(dt.datetime(2019,8,9,15,1,1),dt.datetime(2019,8,10,15,1,1),dico_night_hour) #-1
#day_duration(dt.datetime(2019,8,9,15,0,0),dt.datetime(2019,8,9,16,0,0),dico_night_hour) #3600
#day_duration(dt.datetime(2019,8,9,15,0,0),dt.datetime(2019,8,9,16,0,1),dico_night_hour) #-1


def name_level(x,dico_night_hour):
    m = correct_key(x,dico_night_hour)
    return(dt.datetime(1,1,1,x.hour,x.minute,0)>=dt.datetime(1,1,1,dico_night_hour[m]['end_day_h'],
                                                     dico_night_hour[m]['end_day_m'],0))    
    
    
    
def HenVariable(df_ts, config, ts_name, name_='', save=True, compute_chi2_distance=True, time4entropy=True):
    
    ''' Note: work with ts that have nan (typically at begining)
    Compute some variable at a "level" level, which could be from 17h-3h i.e. over two consecutive days.
    We assume that the df_ts has equally spaced timestamp values and that nbr_sec is the duration in between each timestamp (by construction should be the case)
    
    Input:
    df_ts: Each row correspond to a specific timestamp, each column to a specific hen timeseries (which column name must start 
        with hen_ ). Must also have a Timestamp and a level column, which will be used to aggregate info and compute variables on these 
        aggregated info
    config: file with parameter
    
    Output:
    daily dataframe (where daily is according to the level variable) with according variables'''
    
    #start recording the time it last
    START_TIME = time.clock()
    
    ############ initialise parameters from config file
    nbr_sec = config.nbr_sec
    path_extracted_data = config.path_extracted_data
    li_date2remove = config.li_date2remove
    id_run = config.id_run
    dico_night_hour = config.dico_night_hour
    dico_zone_order = config.dico_zone_order
    li_day_hours = config.li_day_hours
    ValueDelta = config.ValueDelta
    EntropyTimeComputation = config.EntropyTimeComputation
    NbrData = config.NbrData
    
    ############ add correct 'level' variable (i.e. consecutive time slot for night time series)
    df_ts['is_day'] = df_ts['Timestamp'].map(lambda x: is_day(x, dico_night_hour))
    #note that minuit is: 0, and its date should be as 1,2 (day-1, day)
    if ts_name == 'time_serie_night':
        df_ts = df_ts[~df_ts['is_day']].copy()
        df_ts['level'] = df_ts['Timestamp'].map(lambda x: str(x)[0:-9]+'_'+str(x+dt.timedelta(days=1))[8:10] if\
                                                name_level(x,dico_night_hour) else str(x-dt.timedelta(days=1))[0:-9]+'_'+str(x)[8:10])
    elif ts_name == 'time_serie_day':
        df_ts = df_ts[df_ts['is_day']].copy()
        df_ts['level'] = df_ts['day'].copy()
    else:
        print('ERROR: ts_name parameter must either be time_serie_night or time_serie_day')
        sys.exit()
        
    ############ verifications
    #verify columns name of df_ts and select the column we need
    li_hen = [i for i in list(df_ts) if i.startswith('hen_')]
    if not all([i in df_ts.columns for i in ['Timestamp','level']]):
        print('ERROR: your df_ts must have timestamp and level column name')
        sys.exit()
    df = df_ts.filter(['Timestamp','level']+li_hen).copy()
    #verify that the timestamp has same difference than the suggested nbr_sec parameter
    df = df.sort_values('Timestamp')
    if (df['Timestamp'].iloc[1]-df['Timestamp'].iloc[0]).seconds!=nbr_sec:
        print('ERROR: your timestamp difference does not equal your nbr_sec parameter')
        sys.exit()
    
    #list of involved level
    li_day = set(df['level'].tolist())  

    ############ one row per unique hen-timestamp 
    df = pd.melt(df, id_vars=['Timestamp','level'], value_vars=li_hen)
    df.rename(columns={'variable':'HenID','value':'Zone'}, inplace=True)
    #we define the duration of each row to be the nbr_sec, its better than computing with the next timestamp as if we removed some days
    #due to health-assessemnt, then it will induce wrong durations! also more efficient that way. BUT its an assumption, that the row must
    #be equally spaced and nbr_sec is the duration in between each timestamp
    df['duration_sec'] = nbr_sec
    #list of not nan Zones
    li_Zone = [x for x in df[~df['Zone'].isnull()]['Zone'].unique()]

    ########################################################
    print('----------------- total duration per Zone....')
    #one row per day, hen, existingzone
    df_ = df.groupby(['HenID','level','Zone'])['duration_sec'].agg(lambda x: sum(x)).reset_index()
    #one row per day and hen, each columns account for a zone_duration
    df_daily = df_.pivot_table(values='duration_sec', index=['HenID', 'level'], columns='Zone')
    df_daily.rename(columns={x:'duration_'+x for x in li_Zone}, inplace=True)
    #lets verify with total duration
    df_daily['verification_daily_total_duration'] = df_daily.apply(lambda x: np.nansum([x[i] for i in ['duration_'+x for x in li_Zone]]),
                                                                   axis=1)
    df_daily = df_daily.reset_index()
    #replace np.nan duration by 0
    df_daily.replace(np.nan,0, inplace=True)
    df_daily['verification_daily_total_nbr_hour'] = df_daily['verification_daily_total_duration'].map(lambda x: x/60/60)
    print('The number of hours per \"level\" period is of:')
    display(df_daily.groupby(['verification_daily_total_nbr_hour'])['level','HenID'].agg(lambda x: list(x)).reset_index())

    #create an ordered list of the normalized duration per zone for chi2distance later (hen will first be sorted by entropy, and 
    #hence we will do this at the end)
    li_zone_dur = [c for c in df_daily.columns if c.startswith('duration_')] #keep same order
    df_daily['dur_values'] = df_daily.apply(lambda x: str([x[i] for i in li_zone_dur]), axis=1)
    df_daily['dur_values'] = df_daily['dur_values'].map(lambda x: eval(x))
    df_daily['dur_values_normalized'] = df_daily['dur_values'].map(lambda x: [i/float(np.sum(x)) if float(np.sum(x))!=0 else 0 for i in x])
    
    ########################################################
    print('----------------- first time stamp in each zone per day....')
    df_ = df.groupby(['HenID', 'level','Zone'])['Timestamp'].agg(lambda x: min(list(x))).reset_index()
    #agg function = 'first' ats its string value, and the default function is the mean. Here by construction df_ has unique such 
    #values
    df__ = df_.pivot_table(values='Timestamp', index=['HenID', 'level'], columns='Zone', aggfunc='first')
    df__.rename(columns={x:'FirstTimestamp_'+x for x in li_Zone}, inplace=True)
    df__ = df__.reset_index()
    df_daily = pd.merge(df_daily, df__, how='outer', on=['HenID','level'])

    ########################################################
    print('----------------- number of Zone (excluding nan)....')
    df_ = df[~df['Zone'].isnull()].groupby(['HenID','level'])['Zone'].agg(lambda x: len(set((x)))).reset_index()
    df_.rename(columns={'Zone':'Total_number_zone'}, inplace=True)
    df_daily = pd.merge(df_daily, df_, how='outer', on=['HenID','level'])

    ########################################################
    ####running SampEnt, DistrEnt computed over the whole period and not only the day. taking only the value at 17h
    #faster thanks to ValueDelta
    if time4entropy:
        print('----------------- Running entropies at end of each level....')
        dico_HenID_day_ent = {}
        for k, df_hen in df.groupby(['HenID']):
            df_hen = df_hen[~df_hen['Zone'].isnull()]
            df_hen['Zone'] = df_hen['Zone'].map(lambda x: dico_zone_order[x])
            dico_HenID_day_ent[k] = {}
            for L in df_hen['level'].unique():
                df_ = df_hen[df_hen['level']<=L]
                ts_value = df_.tail(1)['Timestamp'].values[0]
                li_zone = df_['Zone'].tolist()
                #restrict the time serie to one value per ValueDelta seconds
                li_zone = [x for i,x in enumerate(li_zone) if i%ValueDelta==0]
                nbr_value = len(li_zone)
                dico_HenID_day_ent[k][pd.to_datetime(L)] = {'SampEnt': sample_entropy(li_zone, order=2, metric='chebyshev'),
                                                            'DistEnt': DistributionEntropy(li_zone), 
                                                            'ts_value': ts_value, 'nbr_value': nbr_value}
                for zone_ in dico_zone_order.values():
                    dico_HenID_day_ent[k][pd.to_datetime(L)]['SampEnt_'+str(zone_)] = sample_entropy([int(z==zone_) for z in li_zone],
                                                                                                     order=2, metric='chebyshev')
        df_daily['RunSampEnt_onLastTsOfEachLevel'] = df_daily.apply(lambda x: dico_HenID_day_ent[x['HenID']][x['level']]['SampEnt'],
                                                                    axis=1)
        df_daily['RunDistEnt_onLastTsOfEachLevel'] = df_daily.apply(lambda x: dico_HenID_day_ent[x['HenID']][x['level']]['DistEnt'],
                                                                axis=1)
        df_daily['RunEnt_onLastTsOfEachLevel_nbr_value'] = df_daily.apply(lambda x: dico_HenID_day_ent[x['HenID']][x['level']]
                                                                              ['nbr_value'], axis=1)
        df_daily['RunEnt_onLastTsOfEachLevel_ts_value'] = df_daily.apply(lambda x: dico_HenID_day_ent[x['HenID']][x['level']]
                                                                             ['ts_value'], axis=1)    
        for zone_ in dico_zone_order.values():
            df_daily['RunSampEnt_onLastTsOfEachLevel_'+str(zone_)] = df_daily.apply(lambda x: dico_HenID_day_ent[x['HenID']][x['level']]['SampEnt_'+str(zone_)],axis=1)   
        
    
    ########################################################        
    #compute some variables based on a list of zones over a day, where each zone count for the same nbr_sec second
    #e.g.[einstreu,eintreu,rampe,rampe.....]
    #excluding empty zones, because it influences for exemple the entropy computation (if full of nan, then might be more predictable)    
    print('----------------- compute some variables based on a list of zones over a day....')
    #minimum/maximum/median duration in a zone and its list, number of transition, entropy,
    #number of bouts spent per zone per level. Note that a transition from one zone x to another zone x (same zone), then it will 
    #be counted as only one bout
    #shifted entropy: only per day. distribtion for all-zones. sampent for each zones and all-zones
    #running entropy: can be per day but for now we keep it over the whole period
    #Note: SampEnt_perZone takes 2 hours for ~50 hen during 6 days
    function2apply = {'list_of_durations': lambda x: list_of_durations(x, nbr_sec),
                      'zone_list': lambda x: tuple(x),
                      'Max_duration_zones': lambda x: max_duration_zones(x),
                      'Max_duration': lambda x: max_duration(x, nbr_sec),
                      'Min_duration': lambda x: min_duration(x, nbr_sec),
                      'Median_duration': lambda x: median_duration(x, nbr_sec),
                      'Average_duration': lambda x: average_duration(x, nbr_sec),
                      'Variance_duration': lambda x: var_duration(x, nbr_sec),
                      'dico_zone_duration':lambda x: dico_zone_duration(x, nbr_sec),
                      'Total_number_transition': lambda x: nbr_transition(list((x))),
                      'nbr_bouts': lambda x: nbr_bouts_per_zone(list((x))),
                      'distribution_entropy': lambda x: DistributionEntropy(list(x))}
                      #'SampEnt_order2': lambda x: sample_entropy([dico_zone_order[i] for i in x], order=2, metric='chebyshev'),
                      #'SampEnt_order3': lambda x: sample_entropy([dico_zone_order[i] for i in x], order=3, metric='chebyshev'),
                      #'SampEnt_perZone': lambda x: {k: sample_entropy([int(i==k) for i in x], order=2, 
                      #                                                metric='chebyshev') for k in dico_zone_order.keys()} }
                      #'shifted_SampEnt_Stat': lambda x: stats_on_shifted_SampEnt(list(x), NbrData, ValueDelta,
                      #                                                                 EntropyTimeComputation),
                      #'shifted_SampEnt_Stat_perZone': lambda x: {k: stats_on_shifted_SampEnt([int(i==k) for i in x], NbrData, 
                      #                                 ValueDelta, EntropyTimeComputation) for k in dico_zone_order.keys()},
                      #'shifted_DistEnt_Stat': lambda x: stats_on_shifted_DistEnt(list(x), NbrData, ValueDelta,
                      #                                                                 EntropyTimeComputation)}
    df_ = df[~df['Zone'].isnull()].groupby(['HenID','level'])['Zone'].agg(function2apply).reset_index()
    df_daily = pd.merge(df_daily, df_, how='outer', on=['HenID','level'])
    for z in li_Zone:
        df_daily['nbr_bouts_'+z] = df_daily['nbr_bouts'].map(lambda x: x.get(z,0))
    df_daily.drop(['nbr_bouts'], inplace=True, axis=1)
    #add maximum duration in zone4
    df_daily['Max_duration_zone_4'] = df_daily['dico_zone_duration'].map(lambda x: max(x.get('zone_4',[0])))
    if time4entropy:
        df_daily['SampEnt_order2'] = df_daily['zone_list'].map(lambda x: sample_entropy([dico_zone_order[i] for i in x], order=2,
                                                                                        metric='chebyshev'))
        df_daily['SampEnt_perZone'] = df_daily['zone_list'].map(lambda x: {k: sample_entropy([v==k for i,v in enumerate(x) if i%ValueDelta==0], order=2, metric='chebyshev') for k in dico_zone_order.keys()})  

    
    ########################################################
    ####Chi2distance (computed here as its a variable and not a visual computed out of variable (plus we need to have ts_name
    #for saving))
    #sort by lowest entropy (i.e. need less info to predict futur. more predictibale should induce more similarities 
    #(as less different solution)), to potentialy make a nicer to look visual
    df_ = df_daily.groupby(['HenID'])['distribution_entropy'].agg(lambda x:np.mean(x)).reset_index().sort_values(['distribution_entropy'])
    li = df_['HenID'].tolist()
    axis_label = [i.split('_')[1] for i in li]
    
    #create path to save visual if not existing
    path_ = os.path.join(path_extracted_data,'visual','chi2distance')
    #create a director if not existing
    if not os.path.exists(path_):
        os.makedirs(path_)
        
    #compute for every level a symmetric chi2distance heatmap
    if compute_chi2_distance:
        print('----------------- Compute Chi2 distance....')
        li_date = [i for i in df_daily['level'].unique() if i not in li_date2remove]
        for d in li_date:
            M = np.zeros(shape=(len(li),len(li)))
            for i, h1 in enumerate(li[:-1]):
                for j in range(i+1,len(li)):
                    h2 = li[j]
                    li_hen_in = df_daily[df_daily['level']==d]['HenID'].unique()
                    #if both hen have at least one record this day (typically not always all hen have values the first day of session)
                    if (h1 in li_hen_in) & (h2 in li_hen_in):
                        l1 = df_daily[(df_daily['HenID']==h1)&(df_daily['level']==d)]['dur_values_normalized'].values[0]
                        l2 = df_daily[(df_daily['HenID']==h2)&(df_daily['level']==d)]['dur_values_normalized'].values[0]
                        chi2 = chi2_distance(l1,l2)
                        M[i][j] = chi2
                        M[j][i] = chi2
            #clear old plot
            plt.figure()
            sns.set(font_scale=0.25) 
            sns.heatmap(M, cmap="YlGnBu", xticklabels=axis_label, yticklabels=axis_label)
            plt.title(d)
            if save:
                plt.savefig(os.path.join(path_,id_run+'_chi2distance_'+str(d).split('T')[0]+'_'+ts_name+'.png'), dpi=300,
                            format='png',bbox_inches='tight')
            #plt.show() 
            plt.close()
    
    #save
    if save:
        df_daily.drop(['verification_daily_total_nbr_hour'],inplace=True,axis=1) #verification_daily_total_duration
        df_daily.to_csv(os.path.join(path_extracted_data, id_run+'_'+ts_name+'_'+name_+'_variables.csv'), sep=';', index=False)

    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))
        
    return(df_daily)


def heatmap_duration_perzone_perhen(df_daily, config, save=True):

    '''heatmap of duration per zone per day per bird'''

    #initialize parameters
    path_extracted_data = config.path_extracted_data
    id_run = config.id_run
    
    #create path where to save if not existing yet
    path_ = os.path.join(path_extracted_data,'visual','duration_per_zone')
    #create a director if not existing
    if not os.path.exists(path_):
        os.makedirs(path_)
    #for each day draw a heatmap
    li_duration = [i for i in df_daily.columns if i.startswith('duration_')]
    for day in tqdm.tqdm(df_daily['level'].unique()):
        #clear old plot
        plt.figure()
        df_ = df_daily[df_daily['level']==day][li_duration+['HenID']]
        df_ = df_.set_index('HenID').transpose()
        sns.set(font_scale=0.3);
        sns.heatmap(df_,cmap="YlGnBu", yticklabels=df_.index, xticklabels=df_.columns);
        
        #save
        if save:
            plt.savefig(os.path.join(path_, id_run+'_duration_per_zone_'+str(day).split(' ')[0].split('T')[0]+'.png'), dpi=300,
                        format='png',bbox_inches='tight')
        plt.show()
        plt.close()
        
def boxplot_distribution_entropy(df_daily, ts_name, config, save=True):
    
    '''Boxplot of daily entropy'''
    
    #initialize parameters
    path_extracted_data = config.path_extracted_data
    id_run = config.id_run
    
    #to add penid to do plot per pen: one must register a hen csv with henid and penid columns, open it here and join it
    
    #distribution entropy visual
    df_ = df_daily.copy()
    df_['level'] = df_['level'].astype(str) # in order to remove hour second minute
    #add one empty level to have names hen at the end
    df_ = df_.reset_index()
    df_.loc[df_.shape[0], 'level'] = ''
    #choose color #TODO: modify to unique color!!!!
    N = len(df_['HenID'].unique())
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    
    #plot
    fig, ax = plt.subplots(figsize=(30,30)) 
    ax = sns.boxplot(x='level', y='distribution_entropy', data=df_, palette="PRGn")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, size=20);
    sns.set(font_scale=1)  #hen names
    ax = sns.swarmplot(x="level", y="distribution_entropy", data=df_, hue="HenID", size=8, palette=RGB_tuples)
    plt.title('Distribution entropy per level across all hen', size=30)
    if save:
        plt.savefig(os.path.join(path_extracted_data,'visual',id_run+'_'+ts_name+'_distribution_entropy.png'),dpi=300,format='png',
                    bbox_inches='tight')
    plt.show()
    plt.close()

    
#Strongly inspired from: https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
def crosscorr(datax, datay, shift=0, wrap=False):
    """ shift-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    shift : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if datax.shape[0]!=datay.shape[0]:
        print('ERROR: the two pandas.Series must be of equal length')
        sys.exit()
    #TODO: WHAT IS WRAP?
    if wrap:
        shiftedy = datay.shift(shift)
        shiftedy.iloc[:shift] = datay.iloc[-shift:].values
        return datax.corr(shiftedy)
    else: 
        #first we shift the pandas series by adding nan at the begining and removing the last values, keeping the same length
        return datax.corr(datay.shift(shift))

    
def CrossCorrelationHeatmap(df_ts, ValueDelta, window_size, step_size, shift, name_, config, save = True, ts_name = 'time_serie_day',
                           li_choosen_relationship=[], if_not_already_done=True):
    #open time series per session and compute the variables for each session (car time series make sence at session level), et en 
    #plus des variables tel que running entropy over the whole session ateach last timestamp of each level make sence only at 
    #session level
    
    #initialize var
    dico_zone_order = config.dico_zone_order
    path_extracted_data = config.path_extracted_data
    dico_night_hour = config.dico_night_hour
    id_run = config.id_run

    ############ add correct 'level' variable (i.e. consecutive time slot for night time series)
    df_ts['is_day'] = df_ts['Timestamp'].map(lambda x: is_day(x, dico_night_hour))
    #note that minuit is: 0, and its date should be as 1,2 (day-1, day)
    if ts_name == 'time_serie_night':
        df_ts = df_ts[~df_ts['is_day']].copy()
        df_ts['level'] = df_ts['Timestamp'].map(lambda x: str(x)[0:-9]+'_'+str(x+dt.timedelta(days=1))[8:10] if\
                                                name_level(x,dico_night_hour) else str(x-dt.timedelta(days=1))[0:-9]+'_'+str(x)[8:10])
    elif ts_name == 'time_serie_day':
        df_ts = df_ts[df_ts['is_day']].copy()
        df_ts['level'] = df_ts['day'].copy()
    else:
        print('ERROR: ts_name parameter must either be time_serie_night or time_serie_day')
        sys.exit()

    ######################### rolling window time lagged cross correlation
    path_ = os.path.join(path_extracted_data,'visual','follower_followed')
    if not os.path.exists(path_):
        os.makedirs(path_)
    li_hen = [x for x in df_ts.columns if x.startswith('hen_')]
    l = len(li_hen)
    M = np.zeros(shape=(l,l))
    print('There is %d hens and hence %d relation'%(l, (l*l-l)/2))
    #print('The length of the TS considered was %d and after removing some value (Valuedelta) we end up %d'%(df_ts.shape[0],
    #                                                                                                        len(d1)))
    for i, h1 in enumerate(li_hen[:-1]):
        d1 = pd.Series([dico_zone_order.get(x, np.nan) for i,x in enumerate(df_ts[h1]) if i%ValueDelta==0])
        for j in range(i+1,l):
            h2 = li_hen[j]
            p = []
            path_plt = os.path.join(path_, id_run+'_cross_correlation_'+name_+'_'+h1+'&'+h2+'_'+ts_name+'.png')
            if if_not_already_done:
                p = glob.glob(path_plt)
            if len(p)==1:
                continue
            if ((len(li_choosen_relationship)>0) & ({h1,h2} in li_choosen_relationship)) | (len(li_choosen_relationship)==0):
                
                d2 = pd.Series([dico_zone_order.get(x, np.nan) for i,x in enumerate(df_ts[h2]) if i%ValueDelta==0])
                #compute
                t_start = 0 ; t_end = t_start + window_size
                rss=[]
                while t_end < d1.shape[0]:
                    d1_ = d1.iloc[t_start:t_end]
                    d2_ = d2.iloc[t_start:t_end]
                    rs = [crosscorr(d1_, d2_, lag, wrap=False) for lag in range(-shift,shift)]
                    rss.append(rs)
                    t_start = t_start + step_size
                    t_end = t_end + step_size
                rss = pd.DataFrame(rss)

                f,ax = plt.subplots(figsize=(10,10))
                #change the limit of the color map
                sns.heatmap(rss,cmap='RdBu_r', ax=ax, vmin=-1, vmax=1)
                ax.set(title=f'Rolling Windowed Time Lagged Cross Correlation', xlim=[0,2*shift], xlabel='Offset',
                       ylabel='Epochs');
                ax.set_xticks(range(0, 2*shift, 20))
                ax.set_xticklabels(range(-shift,shift,20));
                #white mean nan i.e. zero: e.g. one time serie is constant
                if save:
                    plt.savefig(path_plt, dpi=300,format='png',bbox_inches='tight')
                #plt.show()    
                plt.close()

    
    
    
    
    
    