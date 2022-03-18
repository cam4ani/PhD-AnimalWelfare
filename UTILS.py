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
import colorsys
import re
import cv2
import pickle
from operator import itemgetter
import math 
import gc
import uuid #generte random id

#find day of the week
import calendar

#graph analysis
import networkx as nx

#raise warnings
import warnings

#some stats
import scipy
from scipy.stats import entropy, kurtosis, skew, spearmanr, pearsonr, median_abs_deviation

#combination
from itertools import chain, combinations

#time series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf

#PCA
from sklearn.preprocessing import scale
from sklearn import decomposition

#clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import kmodes
from kmodes.kmodes import KModes #with categorical var as well
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
#from skimage.measure import compare_ssim
#other image package
from PIL import Image
import skimage.draw
import imutils

#videos
import imageio
#from skimage import color #kernel die
from colorsys import hls_to_rgb


#models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, cohen_kappa_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#normality test of transitions
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from termcolor import colored

#change r to 0 from 0.2
sys.path.append('C:\\Users\\camil\\Desktop\\animals_code\\entropy')
#from entropy.entropy import sample_entropy #DEPRECIATED use now antropy!

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

#from python documentation
def findDay(date): 
    '''input: dt.datetime(2020,9,30)
       output the week day (e.g. wednesday)'''
    born = date.weekday() 
    return (calendar.day_name[born]) 

def rainbow_color_stops(n=10, end=2/3):
    '''tken frm internet: list of distinct color in gradient of form: (1.0, 0.6666666666666666, 0.0)'''
    return [ hls_to_rgb(end * i/(n-1), 0.5, 1) for i in range(n) ]


def ShapiroTest(data, alpha=0.05):
    '''The tests assume that that the sample was drawn from a Gaussian distribution, and reject hypothesis if not drawn from a normal distribution'''
    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print(colored('Sample looks Gaussian (fail to reject H0)','green'))
    else:
        print(colored('Sample does not look Gaussian (reject H0)','red'))
    return(stat, p)
    
def AgostinosK2Test(data, alpha=0.05):
    '''The tests assume that that the sample was drawn from a Gaussian distribution, and reject hypothesis if not drawn from a normal distribution'''
    stat, p = normaltest(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print(colored('Sample looks Gaussian (fail to reject H0)','green'))
    else:
        print(colored('Sample does not look Gaussian (reject H0)','red'))
    return(stat, p)
        
'''print('ShapiroTest')
stat, p = ShapiroTest(list(df[v].dropna()))
print('D’Agostino’s K^2 Test')
stat, p = AgostinosK2Test(list(df[v].dropna()))

print('------ LOG TRANSFORMATION')
data = np.log(list(df[v].dropna()))
if 0 in list(df[v].dropna()):
    data = np.log(list(df[v].dropna()+1))
print('ShapiroTest')
stat, p = ShapiroTest(data)
print('D’Agostino’s K^2 Test')
stat, p = AgostinosK2Test(data)'''

#from python: https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]
#small examples    
print(most_frequent([2, 1, 2, 2, 1, 3]))


##########################################################################################################################################
################################################ verification of configuration parameters ################################################
##########################################################################################################################################

def config_param_ver(config):
    
    #open param
    dico_zone_order = config.dico_zone_order
    dico_matching = config.dico_matching
    
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
############################################################ Preprocessing ###############################################################
##########################################################################################################################################

def FB_process(config):
    '''minor processing - make it nicer m& more homogeneous with other dataset'''
    #ASSUMPTION: each henID is linked to a unique PenID!
    #some hens appeared several times (e.g. changed tags, changed legrings,...), so first we will join all info
    
    #initialise parameters
    path_FocalBird = config.path_FocalBird
    date_max = config.date_max 
    
    df_FB = pd.read_csv(path_FocalBird, sep=',', parse_dates=['StartDate','EndDate','InitialStartDate'], dayfirst=True, encoding='latin')
    #fill end date to today+1 for the birds which we dont know when is there end date (+1: so that today is taken into account)
    df_FB['EndDate'].fillna(date_max+dt.timedelta(days=1), inplace=True)
    df_FB['TagID'] = df_FB['TagID'].map(int).map(str)  
    df_FB['TagID'] = df_FB['TagID'].map(lambda x: 'tag_'+str(x)) 
    #df_FB['PenID'] = df_FB['PenID'].map(int).map(str)
    #df_FB['PenID'] = df_FB['PenID'].map(lambda x: 'pen'+str(x))
    df_FB['HenID'] = df_FB['HenID'].map(int).map(str)
    df_FB['HenID'] = df_FB['HenID'].map(lambda x: 'hen_'+str(x)) 

    #remove typos
    df_FB['FocalLegringName'] = df_FB['FocalLegringName'].map(lambda x: x.strip(' '))
    #we cant remove the disfunctioning tags now otherwise we will lose theire weight for future
    return df_FB

def FB_process_hen(config):
    '''create one row per hen'''
    #we cant use FB_process as it removes the animals that have no tracked days, whereas we might want to keep for HA
    df_FB = FB_process(config)
    df_FB = df_FB.fillna('')
    #process weight
    li_weight = [x for x in df_FB.columns if 'weight' in x]
    df_FB[li_weight] = df_FB[li_weight].applymap(lambda x: str(x).replace(',','.'))
    li_var_hen = ['HenID','PenID','CLASS','R-Pen','InitialStartDate','early_death']
    df_FB_ = df_FB.groupby(['HenID'])[[i for i in li_var_hen if i!='HenID']+li_weight].agg(lambda x: list(set([i for i in x if i!='']))).reset_index()
    li = [i for i in df_FB_.columns if i not in ['HenID', 'CLASS']]
    df_FB_['CLASS'] = df_FB_['CLASS'].map(lambda x: x[0] if len(x)==1 else None)
    df_FB_[li] = df_FB_[li].applymap(lambda x: x[0] if len(x)==1 else np.nan)
    df_FB_[li_weight] = df_FB_[li_weight].applymap(lambda x: float(x) if '+70 -30' not in str(x) else np.nan)
    df_FB_['Treatment'] = df_FB_['PenID'].map(lambda x: 'OFH' if x in ['pen3','pen5','pen9','pen11',3,5,9,11] else 'TRAN')
    df_FB_['early_death'] = df_FB_['early_death'].fillna(0).replace(2,1) #2 means not sure
    #save as a focalbirds csv
    df_FB_.to_csv(os.path.join(config.path_extracted_data, config.id_run+'df_FOCALBIRDS.csv'), sep=';', index=False)
    return df_FB_


def FB_daily(config):
    '''will be used for example in preprocessing_Origins in order to match each (tagid,date) to a unique henid'''
    
    #download process Focalbirds info
    df_FB = FB_process(config)
    
    #exclude rows were tags were not functionning correctly for some reason 
    df_FB = df_FB[df_FB['ShouldBeExcluded']!='yes']
    #define a list with the active tags of today/last date
    li_active_tags = list(df_FB[df_FB['EndDate']>=config.date_max]['TagID'].unique())
    #Counter(li_active_tags)
    print('From the focalBirdinfo, you have %d ative tags'%len(li_active_tags))

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
    df_FB_daily['date'] = df_FB_daily['date'].map(lambda x: dt.datetime(x.year,x.month,x.day))
    return(df_FB_daily)


def clean_device(config, df_device):
    df_device_ = df_device[df_device['sort']=='Tag'].copy()
    df_device_['date'] = df_device_['Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day))
    df_device_['is_day'] = df_device_['Timestamp'].map(lambda x: is_day(x, config.dico_night_hour))
    df_device_ = df_device_[df_device_['is_day']]
    df_device_ = df_device_.reset_index(drop=True)
    print(df_device_.shape)
    df_device_.head(3)

    ########## compute the gap since the actual time and last action: replace np.nan with 999999
    #if lastaction is nan, it will return nan
    df_lastactionGAP = df_device_.reset_index(drop=True).copy()
    df_lastactionGAP['gap'] = df_lastactionGAP.apply(lambda x: (x['Timestamp']-x['LastAction']).total_seconds(), axis=1)
    df_lastactionGAP = df_lastactionGAP.groupby(['sender','date'])['gap'].agg(lambda x: sorted(list(x))).reset_index()
    df_lastactionGAP['bigest_gap'] = df_lastactionGAP['gap'].map(lambda x: max(x))
    df_lastactionGAP['nbr_obs'] = df_lastactionGAP['gap'].map(lambda x: len(x))
    df_lastactionGAP['bigest_gap'] = df_lastactionGAP['bigest_gap'].fillna(999999)
    df_lastactionGAP['sender'] = df_lastactionGAP['sender'].map(lambda x: 'tag_'+str(int(x)))
    #df_test = df_lastactionGAP.groupby('date')['nbr_obs'].agg(lambda x: Counter(x)).reset_index()
    #df_test['nbr_possibility'] = df_test['nbr_obs'].map(lambda x: len(x))
    #df_test['value'] = df_test['nbr_obs'].map(lambda x: list(x.keys())[0])
    #df_test[df_test['nbr_possibility']>1]

    ######### #LFCOUNTER==0
    df_LFCoutner = df_device_.groupby(['sender','date'])['LFCounter'].agg(lambda x: sorted(list(x))).reset_index()
    #count nan as beeing 0
    df_LFCoutner['LFCounter_nbr_equal0'] = df_LFCoutner['LFCounter'].map(lambda x: sum([((i==0)|(math.isnan(i)==True)) for i in x]))
    #df_LFCoutner['LFCounter_atleastone0'] = df_LFCoutner['LFCounter_nbr_equal0'].map(lambda x: x>0)
    print(df_LFCoutner.shape)
    df_LFCoutner['sender'] = df_LFCoutner['sender'].map(lambda x: 'tag_'+str(int(x)))

    ######### merge both
    df_alldeviceinfo = pd.merge(df_LFCoutner, df_lastactionGAP, on=['sender','date'], how='outer')
    print(df_alldeviceinfo.shape)
    #df_alldeviceinfo.head(3)

    df_alldeviceinfo['expected_nbr_obs'] = df_alldeviceinfo['date'].map(lambda x: nbr_device_info(x))
    df_alldeviceinfo['is_nbr_obs_correst'] = df_alldeviceinfo.apply(lambda x: (x['nbr_obs'] in x['expected_nbr_obs']) | (x['nbr_obs']>max(x['expected_nbr_obs'])), axis=1)
    df_alldeviceinfo['2keep'] = df_alldeviceinfo.apply(lambda x: (x['bigest_gap']<=(12*60)) &\
                                                       (x['LFCounter_nbr_equal0']<config.lf_counter)&\
                                                       (x['is_nbr_obs_correst']==True),axis=1)
    #df_alldeviceinfo['2keep'].value_counts()
    #save
    df_alldeviceinfo.to_csv(os.path.join(config.path_extracted_data, config.id_run+'_df_alldeviceinfo.csv'),sep=';')
    return df_alldeviceinfo


def preprocessing_Origins(paths, config, save=True, dodevice=True):
    
    '''Each experiment should have his own preprocessing function
    It opens from a list of csv-path all the csv and aggregated them and put into correct format
    output one dataframe'''

    ####################################################################################
    ############################### Initialise variables ###############################
    ####################################################################################
    path_extracted_data = config.path_extracted_data
    id_run = config.id_run   
    dico_matching = config.dico_matching
    starting_date = config.starting_date
    
    #create path to save extracted data/info if not existing
    if not os.path.exists(path_extracted_data):
        os.makedirs(path_extracted_data)
        
    ####################################################################################
    ####### Download all logs one by one adding the logfilename and the ts_order #######
    ####################################################################################
    li_df = []
    for path_ in paths:
        df = pd.read_csv(path_, sep=';',names=['Timestamp','TagSerialNumber','TagID','Zone','systemgantnerid','useless_zone',
                                               'signalstrength','zone2','signalstzone2','zone3','signalstrzone3','zone4','signalstrzone4'],
                        dayfirst=True) 
        df['data_path'] = path_
        df['system'] = path_.split('Barn 4 Pen ')[1].split('\\')[0]
        log_name = path_.split('\\')[-1].split('.')[0]
        df['log_file_name'] = log_name
        df['ts_order'] = df.index.copy() 
        #faster but also works with the fact that soemetimes we have strings at the end of the column than parse_dates
        df['Timestamp'] = df['Timestamp'].map(lambda x: dt.datetime.strptime(x, "%d.%m.%Y %H:%M:%S")) 
        v = df.shape[0]
        print(path_)
        if v<80000:
            print_color((('log: %s has '%log_name,'black'),(v,'red'),(' rows','black')))
        else:
            print_color((('log: %s has '%log_name,'black'),(v,'green'),(' rows','black')))
        li_df.append(df)
    df = pd.concat(li_df)
    
    df = df[~df['TagSerialNumber'].isnull()]
    #### make sure about the type       
    df = df[:-1] #remove last row: often wrong
    df['time'] = df['Timestamp'].map(lambda x: dt.datetime.time(x))
    df['date'] = df['Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day))
    #remove data before the starting date
    df = df[df['date']>=starting_date]
    df['Zone'] = df['Zone'].map(lambda x: x.strip())
    #df['TagID'] = df['TagID'].map(int).map(str)     #ValueError: invalid literal for int() with base 10: '020015BC' sometimes mixed
    df['TagID'] = df['TagID'].map(lambda x: 'tag_'+str(x))
        
    ####################################################################################
    ####################### Add a unique HenID to tracking data ########################
    ####################################################################################  
    df_FB_daily = FB_daily(config)
    #merge tracking data with hens info
    df = pd.merge(df, df_FB_daily, on=['date','TagID'], how='inner') 
    #note that : how=inner in order to only have records that are correctly associated to a chicken
    #how!= left as we need to remove some records if the system was resetting etc, so we dont want to keep the tracking data of tags that were not working correctly on that day

    ####################################################################################
    ############################  match zone to better name ############################
    ####################################################################################      
    #if correct pen, now we can change the names for beeing more userfriendly
    df['Zone'] = df['Zone'].map(lambda x: dico_matching[x])
    
    ####################################################################################
    ###################### create devide output for future needs #######################
    #################################################################################### 
    if dodevice==True:
        print('-------------------- DEVICE DATA --------------------')
        print('process device data...')
        df_device = openDevice(config)
        #it saves in the fct
        df_device_4cleaning = clean_device(config,df_device)
        
    ####################################################################################
    ####################################### Save #######################################
    ####################################################################################   
    df = df.sort_values(['Timestamp'], ascending=True)
    #save with all raw data to use all signal strength and so on for cleaning
    if save:
        df = df.filter(['Timestamp', 'HenID', 'Zone','PenID','log_file_name','ts_order','TagID','signalstrength','system',
                    'time','date','zone2','signalstzone2','zone3','signalstzone3','zone4','signalstzone4']).reset_index(drop=True)
        df.to_csv(os.path.join(path_extracted_data, id_run+'_PreprocessRecords_forcleaning.csv'),sep=';')
    
    return(df)


def openDevice(config):
    '''Very long, so we want to do this only once ideally'''
    
    ####################################################################################
    ############################### Initialise variables ###############################
    ####################################################################################
    path_extracted_data = config.path_extracted_data
    id_run = config.id_run
    path_initial_data = config.path_initial_data
    
    ####################################################################################
    ################################ Open deviceUpdates ################################
    ####################################################################################
    li_logs = []
    for path_system in glob.glob(os.path.join(path_initial_data, 'Barn 4 Pen*\DeviceUpdates')):
        li_ = glob.glob(os.path.join(path_system, 'log*'))
        li_logs.extend(li_) 
    li_newFirmwarecol = ['Timestamp','sort','code','sender','Temperature','Battery Voltage','MovementCounter','LastLFSeen',
                         'LFCounter','LFRSSISum','Zone', 'Subzone','Time','LastInfo','LastLocation','LastAction']
    li_df = []
    for path_log in tqdm.tqdm(li_logs):
        df_device = pd.read_csv(os.path.join(path_log), sep=';', names=li_newFirmwarecol, parse_dates=['LastAction','Timestamp'],
                               dayfirst=True) #veriried dates: correct
        df_device['data_path'] = path_log
        df_device['system'] = 'pens:'+path_log.split('Barn 4 Pen ')[1].split('\\')[0]
        df_device['logID'] = path_log.split('\\')[-1].split('.')[0]
        li_df.append(df_device)
    df_device = pd.concat(li_df)
    df_device = df_device[df_device['Timestamp']>=dt.datetime(2020,9,29)]
    df_device['date'] = df_device['Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day))
    df_device['sender'].fillna(' ', inplace=True)

    return df_device



##########################################################################################################################################
################################################# chapter 0 - testing cleaning methods  ##################################################
##########################################################################################################################################

def cleaning_processing(date_min, date_max, config, df=pd.DataFrame()):
    
    #initialisation
    path_extracted_data = config.path_extracted_data
    id_run = config.id_run
    dico_matching = config.dico_matching
    
    #open the data: for chapter 0: cleaning
    if df.shape[0]==0:
        df = pd.read_csv(os.path.join(path_extracted_data, id_run+'_PreprocessRecords_forcleaning.csv'), sep=';', 
                         parse_dates=['Timestamp', 'date'], index_col=0) 
        df['Zone'] = df['Zone'].map(lambda x: dico_matching[x])
        df['HenID'] = df['HenID'].map(lambda x: 'hen_'+str(x))
        df['TagID'] = df['TagID'].astype(int).astype(str)    
        df['TagID'] = df['TagID'].map(lambda x: 'tag_'+str(x))
        df['PenID'] = df['PenID'].map(lambda x: 'pen'+str(x))

    #filter by the dates we want
    print(df.shape)
    df = df[(df['Timestamp']<date_max)&(df['Timestamp']>date_min)]
    print(df.shape)
        
    #add previous/next zones
    li_df = []
    #more efficient to do it per hen, as it wont need to search in the whole dataframe, and we can simply shift the timestamp column
    for i, df_hen in tqdm.tqdm(df.groupby(['HenID'])):
        #as sometimes before the end of the experiment we sometimes have only one record
        if df_hen.shape[0]>10:
            #as the next record date (sort by date, then simply shift by one row and add nan at then end)
            df_hen = df_hen.sort_values(['Timestamp'], ascending=True)
            #same date, one must take the last recorded one & sorting by date might change it. Also it already should be sorted by date
            for r in range(1,6):
                df_hen['next'+str(r)+'_record_date'] = df_hen['Timestamp'].tolist()[r:]+[pd.NaT for v in range(1,r+1)]

            #compute duration 
            df_hen['duration'] = df_hen.apply(lambda x: (x['next1_record_date']-x['Timestamp']).total_seconds(), axis=1)

            #compute previous and next duration
            df_hen['previous_duration'] = [0]+df_hen['duration'].tolist()[0:-1]
            df_hen['next_duration'] = df_hen['duration'].tolist()[1:]+[0]

            #compute previous record date
            for r in range(1,6):
                df_hen['previous'+str(r)+'_record_date'] = [pd.NaT for v in range(1,r+1)]+df_hen['Timestamp'].tolist()[0:-r]

            #add next & previous record zone
            for r in range(1,6):
                df_hen['next'+str(r)+'_zone'] = df_hen['Zone'].tolist()[r:]+[np.nan for v in range(1,r+1)]
                df_hen['previous'+str(r)+'_zone'] = [np.nan for v in range(1,r+1)]+df_hen['Zone'].tolist()[0:-r]
            #is next zone==previous zone
            df_hen['previousZone==NextZone'] = df_hen.apply(lambda x: x['next1_zone']==x['previous1_zone'], axis=1)

            li_df.append(df_hen)
    #put again in one dataframe
    df = pd.concat(li_df)
    df.sort_values(['Timestamp'], ascending=True, inplace=True)
    df.drop_duplicates(subset=['Timestamp','TagID'], keep='last', inplace=True)
    df = df.drop(['log_file_name'], axis=1)
    #df = df.drop(['ts_order','log_file_name','time'], axis=1)

    #further cleaning
    #match to the correct zone, from the Versuch document
    dico_zones = {1:'2_Zone',
                 2:'3_Zone',
                 3:'4_Zone',
                 4:'5_Zone',
                 5:'1_Zone',
                 6:'miniZone3',
                 7:'miniZone3',
                 8:'miniZone3',
                 9:'miniZone3',
                 0:'NoOtherZone'}
    #except in pens 10-12 where zone 9 is minimarker of tier1 i.e. zone 2!
    li_match = ['zone2','zone3','zone4']
    for c in li_match:
        df[c+'_match'] = df[c].map(lambda x: dico_zones.get(x,np.nan))
    #except in pens 10-12 where zone 9 is minimarker of tier1 i.e. zone 2!
    for c in li_match:
        df.loc[(df['system']=='10 - 12')&(df[c]==9), c+'_match'] = '2_Zone'
    #small verification before droping
    print('small VERIFICATION of zone matching, before droping the old names')
    display(df[['system','zone2','zone3','zone4','zone2_match','zone3_match','zone4_match']].head(10))
    print('specifically, a small verification of when zone2 was equal to 9')
    display(df[df['zone2']==9].head(3))
    df.drop(['zone2','zone3','zone4'], axis=1, inplace=True)

    #add other variables
    df['zone3_match_exist'] = df['zone3_match'].map(lambda x: int(x!='NoOtherZone'))
    df['next2zone==Zone'] = df.apply(lambda x: int(x['next2_zone']==x['Zone']), axis=1)
    df['previous2zone==Zone'] = df.apply(lambda x: int(x['previous2_zone']==x['Zone']), axis=1)
    
    return df    



##########################################################################################################################################
############################################################# time series ################################################################
##########################################################################################################################################


def time_series_henColumn_tsRow(df, config, col_ts='Zone' , name_='', ts_with_all_hen_value=False, save=True, 
                                hen_time_series=False, path_extracted_data=None):
    
    '''one time series with each column being one hen. because then opening one file we have all. also, no need to go column by column to change day'''
    
    #start recording the time it last
    START_TIME = time.perf_counter()        

    #initialize parameter
    if path_extracted_data==None:
        path_extracted_data = config.path_extracted_data
    id_run = config.id_run
    nbr_sec = 1 #should stay one
    
    #create a director if not existing
    if not os.path.exists(path_extracted_data):
        os.makedirs(path_extracted_data)
        
    #verify columns name if not done before: TODO
    
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
    #remove timestamp without all hen, if requested
    if ts_with_all_hen_value:
        df_hens['nbr_nan'] = df_hens.isnull().sum(axis=1)
        #plt.plot(df_hens['nbr_nan']);
        print(df_hens['nbr_nan'].unique())
        print('-------------- Lets remove timestamp without all hen')
        #df_hens[10729:]['nbr_nan'].unique() #only 0 after the first one
        df_hens = df_hens[df_hens['nbr_nan']==0]
        df_hens.drop(['nbr_nan'], inplace=True, axis=1)
        print('as we want the time series to start at the same time, we remove the dates without info on each hen, making us start on ',
              df_hens.iloc[0]['Timestamp'])
    
    df_hens['date'] = df_hens['Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day))
    if save:
        print('-------------- Lets save')
        print(df_hens.shape)
        df_hens.to_csv(os.path.join(path_extracted_data,id_run+'_TimeSeries'+str(name_)+'.csv'), sep=';', index=False)

    #one time serie per hen
    if hen_time_series:
        print('-------------- Lets compute individuals seconds time series')
        #create a director if not existing
        path_ts = os.path.join(config.path_extracted_data, 'HeninitialTs')
        if not os.path.exists(path_ts):
            os.makedirs(path_ts)
        #remove the date keep only the timestamp
        df_hens['Timestamp_value'] = df_hens['Timestamp'].map(lambda x: dt.timedelta(hours=x.hour, 
                                                                                       minutes=x.minute, 
                                                                                       seconds=x.second))        
        li_hen = [h for h in df_hens.columns if h.startswith('hen_')]
        for h in li_hen:
            #select the column associated to the hen
            df_per_hen = df_hens[[h,'Timestamp_value','date']].copy()
            df_per_hen[h] = df_per_hen[h].map(lambda x: int(x[5:]) if x!=None else np.nan)
            #pivot, to put the date in column intead of having one row for each timestamp_value per date
            df_per_hen = df_per_hen.pivot(index='Timestamp_value', columns='date', values=h)
            df_per_hen.reset_index(drop=False, inplace=True)
            df_per_hen.to_csv(os.path.join(path_ts, id_run+'_TimeSeries_initial_'+str(name_)+'_'+h+'.csv'), sep=';')
    
    END_TIME = time.perf_counter()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))  
    
    return(df_hens)




##########################################################################################################################################
######################################################### time series analysis ###########################################################
##########################################################################################################################################


def acf_pacf(df_ts, path_save, config, title_='', egg_zone='zone_4', li_hours_to_consider=[2,3,4,5,6,7,8,9,10,11],
             keep_only_certain_hour=False, li_min = [10,20,30,60], do_plot=False):
    '''compute the acf and pacf and plot if. must be a df as in mobility'''
    START_TIME = time.perf_counter()
    id_run = config.id_run
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    li_hen = [x for x in df_ts.columns if x.startswith('hen_')]
    dico_ = {}
    for HenID in tqdm.tqdm(li_hen):
        dico_[HenID] = {}
        title = HenID+' '+title_
        if keep_only_certain_hour:
            title = title+'_h'+'-'.join([str(x) for x in li_hours_to_consider])
            df_ts[HenID] = df_ts.apply(lambda x: x[HenID] if x['hour'] in li_hours_to_consider else 'zone_0', axis=1)
        if do_plot:
            fig, axes = plt.subplots(len(li_min),2, figsize=(18,8))
        for i,mn in enumerate(li_min):
            dico_[HenID][mn] = {}
            value_in_1h = int(60/mn)
            li_density = density_mnlevel(df_ts[HenID].tolist(),mn*60,egg_zone)
            if sum(li_density)>0:
                #save info
                dico_[HenID][mn]['acf'] = list(acf(li_density, nlags=30*value_in_1h))
                dico_[HenID][mn]['abs_confint'] = 1.96/np.sqrt(len(li_density))
                dico_[HenID][mn]['pacf'] = list(pacf(li_density, nlags=30*value_in_1h))
                ###plot
                if do_plot:
                    #acf
                    fig = plot_acf(li_density, lags=30*value_in_1h, alpha=0.01, ax=axes[i,0])
                    axes[i,0].axvline(x=24*value_in_1h, linewidth=2, color='red')
                    axes[i,0].set_title('ACF '+str(mn))
                    plt.tight_layout()
                    #pacf
                    fig = plot_pacf(li_density, lags=30*value_in_1h, alpha=0.01, ax=axes[i,1])
                    axes[i,1].axvline(x=24*value_in_1h, linewidth=2, color='red')
                    axes[i,1].set_xlabel('Lag');
                    axes[i,1].set_title('PACF '+str(mn))
        if do_plot:
            plt.savefig(os.path.join(path_save,id_run+'_'+title.replace(' ','_')+'.png'), format='png')
            plt.close()
    END_TIME = time.perf_counter()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))  
    return dico_


############# acf pacf without the alpha but with our confint
def sign_acf_pacf(d):
    if len(d)==0:
        return {}
    r = {}
    for mn,d_acfpacf in d.items():
        if 'acf' in d_acfpacf:
            #add condition pour stability
            firstindex_unstable = next(x[0] for x in enumerate(list(d_acfpacf['acf'])+[100]) if abs(x[1])>1)
            d_acfpacf['acf'] = d_acfpacf['acf'][0:firstindex_unstable]
            firstindex_unstable = next(x[0] for x in enumerate(list(d_acfpacf['pacf'])+[100]) if abs(x[1])>1)
            d_acfpacf['pacf'] = d_acfpacf['pacf'][0:firstindex_unstable]
            #extract info
            r[mn] = {'acf': [(x,abs(x)-d_acfpacf['abs_confint']) for x in d_acfpacf['acf']], 
                     'pacf': [(x,abs(x)-d_acfpacf['abs_confint']) for x in d_acfpacf['pacf']]}
    return (r)


def most_sign_pacf(d):
    if len(d)==0:
        return ()
    r = {}
    for mn,d_acfpacf_sig in d.items():
        value_in_1h = int(60/mn)
        li_t = d_acfpacf_sig['pacf']
        li_s = [x[1] for x in li_t]
        #most significant value after 10h lag
        m = max(li_s[value_in_1h*10:])
        if m>0:
            lag = li_s.index(m)
            lag_h = lag*mn/60
            sign_level = li_t[lag][1]
            r[mn] = (lag_h, sign_level)
        else:
            r[mn] = (0,0) #put 0 if none are significant
    return r

############# acf pacf with alpha=0.1 in the acf, pacf fct within the  acf_pacf() function
def is_sign(li_v, li_conf):
    r = []
    for i,conf in enumerate(li_conf):
        v = li_v[i]
        if (v>0) & (conf[0]>0):
            r.append(conf[0])
        elif (v<0) &(conf[1]<0):
            r.append(conf[1])
        else:
            r.append(0)
    return r
#small ex
#li_v, li_conf = acf([1,1,1,1,1,2,1,2,1,2,1,2,1,2,2,2,2,2,2,2,1,1,1,1,1,1], nlags=10, alpha=0.01)
#is_sign(li_v, li_conf)
#[(True, 1.0),
# (False, 0),
# (True, 0.10646899021065026),
# (False, 0),
# (False, 0),
# (False, 0),
# (False, 0),
# (False, 0),
# (False, 0),
# (False, 0),
# (False, 0)]

def sign_acf_pacf_withalphapacf(d):
    if len(d)==0:
        return {}
    r = {}
    for mn,d_acfpacf in d.items():
        if 'acf' in d_acfpacf:
            if type(d_acfpacf['acf'][0])==np.ndarray:
                r[mn] = {'acf': is_sign(d_acfpacf['acf'][0], d_acfpacf['acf'][1]),
                         'pacf': is_sign(d_acfpacf['pacf'][0], d_acfpacf['pacf'][1])}
    return (r)

def most_sign_pacf_withalphapacf(d):
    if len(d)==0:
        return ()
    r = {}
    for mn,d_acfpacf_sig in d.items():
        value_in_1h = int(60/mn)
        li = d_acfpacf_sig['pacf']
        li_abs = [abs(x) for x in li]
        #most significant value after 10h lag
        m = max(li_abs[value_in_1h*10:])
        if m!=0:
            lag = li_abs.index(m)
            lag_h = lag*mn/60
            sign_level = li[lag]
            r[mn] = (lag_h, sign_level)
    return r

    
#li_hen = [x for x in df.columns if x.startswith('hen_')]
#for h in li_hen:
#    df[h+'_signif'] = df[h].map(lambda x: sign_acf_pacf(x))
#    df[h+'_most_sig'] = df[h+'_signif'].map(lambda x: most_sign_acf_pacf(x))
       
    
##########################################################################################################################################
############################################################# correlation ################################################################
##########################################################################################################################################
    
    
def corr_from_dep2feature(li_output, li_cont, df_modelling, p_val=0.05, print_=False):
    '''Computing correlation for continuous vs continuous each var from first list to each one of the second'''
    
    df_significance = pd.DataFrame(columns=['var1', 'var2', 'val_spear', 'pval_spear', 'val_pers','pval_pers'])
    #pearson linear, spearman: monotonic relationship
    i = 0 ; k = 0; t = 0 ; n=0
    for x in tqdm.tqdm(li_output):
        for v in li_cont:
            names = [x,v]
            df = df_modelling.dropna(how='any', subset=names).copy()
            rcoeff1, p_value1 = spearmanr(df[x].tolist(), df[v].tolist())
            rcoeff2, p_value2 = pearsonr(df[x].tolist(), df[v].tolist())
            df_significance.loc[n] = names + [rcoeff1, p_value1, rcoeff2, p_value2]
            n = n+1
            if (p_value1 > p_val) & (p_value2 > p_val): #0.1
                #print("->There IS NO statistically significant CORRELATION between %s and %s" %(x,v))
                pass
            elif (p_value1 < p_val) | (p_value2 < p_val):
                if print_:
                    print('----------------')
                    print("->There IS statistically significant CORRELATION between %s and %s" % (x.upper(),v))    
                    print(rcoeff1,p_value1,rcoeff2,p_value2)
                t = t+1
            else:
                print('there might be an issue')
                print(names)
                k = k+1
    print('-------------------------------------------------------------------')
    print('There is %d potential issues, and %d significant correlation out of %d' %(k,t,df_significance.shape[0]))
    return df_significance

    
def corr_from_feature2feature(li, df_modelling, p_val=0.05, print_=False):
    '''Computing correlation for continuous vs continuous for each combination of two variable in the list (named li)'''
    df_feature_feature = pd.DataFrame(columns=['var1', 'var2', 'val_spear', 'pval_spear', 'val_pers','pval_pers'])
    #pearson linear, spearman: monotonic relationship
    i = 0 ; k = 0; t = 0
    n = 0
    p_val = 0.05
    #correlation between the variable itself
    for i in range(len(li)-1):
        for j in range(i+1,len(li)):
            names = [li[i],li[j]]
            df = df_modelling.dropna(how='any', subset=names).copy()
            rcoeff1, p_value1 = spearmanr(df[names[0]].tolist(), df[names[1]].tolist())
            rcoeff2, p_value2 = pearsonr(df[names[0]].tolist(), df[names[1]].tolist())
            df_feature_feature.loc[n] = names + [rcoeff1, p_value1, rcoeff2, p_value2]
            n = n+1
            if (p_value1 > p_val) & (p_value2 > p_val): #0.1
                #print("->There IS NO statistically significant CORRELATION between %s and %s" %(names[0],names[1]))
                pass
            elif (p_value1 < p_val) | (p_value2 < p_val):
                if print_:
                    print('----------------')
                    print("->There IS statistically significant CORRELATION between %s and %s" %(names[0].upper(),names[1]))    
                    print(rcoeff1,p_value1,rcoeff2,p_value2)
                t = t+1
            else:
                print('there might be an issue')
                print(names)
                k = k+1
    print('-------------------------------------------------------------------')
    print('There is %d potential issues, and %d significant correlation out of %d' %(k,t,df_feature_feature.shape[0]))
    return df_feature_feature    
    
     
    
#compute a graph of correlations to make it easier
def correlationGraph(df_corr, path_save_SNA, dico_nodename_attribute={}, name_='', p_val_spear=0.01, p_val_pers=0.01, 
                     condition_type='or'):
    '''From a dataframe of correlation, compute the associated graph, where
    input: dataframe with the following columns: var1, var2, val_spear, pval_spear, val_pers, pval_pers
    link: if two nodes have a significant p-values (p_val in parameters will be used)
    node: each var1, var2
    edge attribute: weight_coeff: correlation coefficient, weight_pval: correlation p-value
    node attribute: type: feature/dependant
    condition_type: can only be "and", or "or", it will raise an error otherwise '''

    #small test
    if condition_type not in ['and','or']:
        print('ERROR: condition_type: can only be "and", or "or"')
        sys.exit()
        
    #initialise graph
    G = nx.Graph() #only one edge between two nodes (otherwise use MultiGraph)

    #add the nodes and its attribute if any was specified
    dico_name_id = {}
    dico_n_a = {}
    if len(dico_nodename_attribute)!=0:
        for i, (n, dico_attribute) in enumerate(dico_nodename_attribute.items()):
            dico_name_id[n] = i
            dico_n_a[i] = dico_attribute.copy()
            dico_n_a[i]['name'] = n
            G.add_node(i) 
    else:
        for i,n in enumerate(list(set(df_corr['var1'].tolist()+df_corr['var2'].tolist()))):
            dico_name_id[n] = i
            G.add_node(i)
            dico_n_a[i] = {}
            dico_n_a[i]['name'] = n
    nx.set_node_attributes(G, dico_n_a)
  
    #add edges
    li_edges = [] 
    for i in range(df_corr.shape[0]):
        x = df_corr.iloc[i]
        n1 = x['var1'] ; n2 = x['var2'] ; pval_spear = x['pval_spear'] ; pval_pers = x['pval_pers']
        #if significant, link it
        if ((pval_spear<=p_val_spear) & (pval_pers<=p_val_pers) & (condition_type=='and')) |\
           (((pval_spear<=p_val_spear) | (pval_pers<=p_val_pers)) & (condition_type=='or')):
            li_edges.append((dico_name_id[n1], dico_name_id[n2], {'pval_spear':float(round(pval_spear,3)),
                                      'pval_pers':float(round(pval_pers,3)), 
                                      'val_spear':float(round(x['val_spear'],3)), 
                                      'val_pers':float(round(x['val_pers'],3))}))
    #add edges with attributes
    G.add_edges_from(li_edges)
    print(len(li_edges))
    #write G (networkX graph) in GEXF format for gephi
    nx.write_gexf(G, os.path.join(path_save_SNA, name_+'_'+condition_type+'_'+'.gexf'))    
    
        

##########################################################################################################################################
###################################################### Linear discriminant analysis ######################################################
##########################################################################################################################################

#From: https://sebastianraschka.com/Articles/2014_python_lda.html
#feature scaling such as [standardization] does not change the overall results of an LDA and thus may be optional

def plot_cov_ellipse(cov, pos, color, nstd=2, ax=None):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    #width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, alpha=0.1, color=color)
    ax.add_artist(ellip)
    return ellip


def plot_scikit_lda(X, y, lda, title, path_save):
    plt.figure(figsize=(14,6))
    ax = plt.subplot(111)
    #'RdBu', 'rocket_r', 'muted', 'Set1'
    #sns.palplot(sns.color_palette("Set1", n_colors=8, desat=.5))
    for label,labelid,color in zip(set(y),range(len(set(y))),sns.color_palette("Set1", len(set(y)))):
        plt.scatter(x=X[:,0][y==label], #first LDA component e.g. X[:,0][y=='hen_100']
                    y=X[:,1][y==label], #second LDA component
                    color=color,
                    alpha=0.6,
                    label=label)
        #add ellipse TODO IF USING IT: check best way to do ellipse
        points = np.array(list(zip(X[:,0][y==label], X[:,1][y==label])))
        #plot_point_cov(points, nstd=3, alpha=0.5, color='green')
        pos = points.mean(axis=0)
        cov = np.cov(points, rowvar=False)
        #cov = lda.covariance_
        #pos = lda.means_[labelid]
        plot_cov_ellipse(cov, pos, color, 2, ax)

    #list of variance explained
    li_var = list(lda.explained_variance_ratio_)
    plt.xlabel('LD1 - percentage of explained variance: '+str(round(li_var[0]*100)))
    plt.ylabel('LD2 - percentage of explained variance: '+str(round(li_var[1]*100)))
    plt.title(title)
    
    #leg = plt.legend(loc='upper right', fancybox=True)
    #leg.get_frame().set_alpha(0.5)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    #remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(path_save, title+'.png'),dpi=300,format='png',bbox_inches='tight')
    #plt.show()
    
def explained_var(lda, nbr_lda_components, path_save):
    li_var = list(lda.explained_variance_ratio_)
    fig = plt.figure(figsize=(10,3.5))
    fig.suptitle('LDA explained variance (red line : %d components)'%nbr_lda_components) 
    plt.subplot(1,2,1)
    plt.plot(np.round(li_var, decimals=4)*100)
    plt.xlabel('number of LDA components')
    plt.ylabel('% of explained variance')
    plt.axvline(x=nbr_lda_components-1, linewidth=1, color='red')
    plt.subplot(1,2,2)
    plt.plot(np.cumsum(np.round(li_var, decimals=4)*100))
    plt.xlabel('number of LDA components')
    plt.ylabel('cumulative % of explained variance')
    plt.axvline(x=nbr_lda_components-1, linewidth=1, color='red')
    plt.ylim(0,105)
    plt.savefig(os.path.join(path_save, 'explained_variance.png'),dpi=300,format='png',bbox_inches='tight')
    plt.show()
    
    
    
##########################################################################################################################################
####################################################### topic modelling on MLPs #########################################################
##########################################################################################################################################

def MLPsWord4lda(documents, c_words, nbr_times):
    '''clean documents'''
    #remove all words appearing less or equal to nbr_times times in the overall time series
    tokens_removed = set(word for word in c_words.keys() if c_words[word]<nbr_times)
    cleaned_documents = {h_day:[word for word in words if word not in tokens_removed] for h_day,words in documents.items()}
    #perhaps later: remove with to much transition words? keep it for now, I dont know what this means
    return(cleaned_documents, tokens_removed)

def word_from_MLP(li, length_words, starting_hour, dico_size):
    '''From a list of zones, output a list of tuples: (word: described by a list of consecutives zones of length "length_words",
    by teh length categories in each of the zone, and by the the starting time of the word (afternoon or morning))'''
    li_dur = list_of_durations(li,1) #keep to one as here we want to use the output as index
    #print(li_dur)
    #keep most general (in case we want to try different way to aggregate it):
    #create a list of tuple of the initial and last index of each word from the li
    li_t = [(sum(li_dur[0:i]), sum(li_dur[0:i+length_words])) for i in range(len(li_dur)-(length_words-1))]
    #convert each word to a tuple: (word: described by a list of consecutives zones of length "length_words", 
    #index of the starting time of the word)
    #[(li[x[0]:x[1]],x[0])  for x in li_t]
    #convert each word to a tuple: (list of the ordered zones, list of their durations, the index of the original list that the word started)
    li_r = [zone_duration(li[x[0]:x[1]],1)+[x[0]+1]  for x in li_t]
    return [str([x[0],[dico_rule_output(i,dico_size) for i in x[1]],int((starting_hour+int(x[2]/60/60))<12)]) for x in li_r]


def list_tuple_zone_dur(li, nbr_sec):
    li_dur = [len(list(x[1]))*nbr_sec for x in itertools.groupby(li)]
    li_zone = [list(x[1])[0] for x in itertools.groupby(li)]
    return list(zip(li_zone, li_dur))

        
        
def word_from_MLP_end_and_begin_atlongDuration(li, nbr_sec_not_transition, starting_hour, dico_size, config, morethan4_in1=False,
                                               nbr_sec=1):
    '''From a list of zones, output a list of tuples: (word: described by a list of consecutives zones , 
    index of the starting time of the word)'''
    
    dico_zone_order = config.dico_zone_order
    #tuple of durations and zones
    t_z_d = list_tuple_zone_dur(li,nbr_sec)
    #add the second each word starts
    t_z_d_s = [(t_z_d[i][0],t_z_d[i][1],sum([v[1] for v in t_z_d[0:i]])+1) for i in range(len(t_z_d))]
    #for verification
    #print(t_z_d_s)
    #substitue first second of the zone into info on afternoon or morning
    t_z_d_s = [(x[0],x[1],int((starting_hour+x[2]/60/60)<12)) for x in t_z_d_s]
    li_mlps = []
    smlp = []
    for x in t_z_d_s:
        if len(smlp)==0:
            #smlp.append((x[0],dico_rule_output(x[1],dico_size),x[2])) #dont care about the duration
            smlp.append((x[0],x[2]))
        elif x[1]<nbr_sec_not_transition:
            smlp.append((x[0],x[2]))
        else:
            smlp.append((x[0],dico_rule_output(x[1],dico_size),x[2]))
            
            #put the zone in a list
            smlp = [[v[0] for v in smlp], smlp[0][1]]
            
            #remove the transitional zones 
            m = [smlp[0][0]]+[smlp[0][i] for i in range(1,len(smlp[0])-1) if not (dico_zone_order[smlp[0][i-1]]-dico_zone_order[smlp[0][i]])==(dico_zone_order[smlp[0][i]]-dico_zone_order[smlp[0][i+1]])]+[smlp[0][-1]]
            
            #for sub-MLPS with strictly more than 4 zones, transform them into a list of zone ordered by their first 
            #occurences in the list only and add the correct end zone (i.e. car large) at the end if not already correct 
            #and add a third variable 'more than 4'(OR 'nbr of zone modulo 3')
            if len(m)>4:
                if morethan4_in1:
                    smlp = [sorted(list(set(m))), 'more_than_4'] #smlp[1],
                else:
                    r = []
                    for i in m:
                        if i not in r:
                            r.append(i)
                    if r[-1]!=m[-1]:
                        r = r+[m[-1]]
                    smlp = [r, smlp[1], 'more_than_4']
            else:
                smlp = [m, smlp[1], '']
            li_mlps.append(smlp)
            smlp = [(x[0],x[2])]
            t = ''
    return [str(x) for x in li_mlps]
#li = [1,1,1,1,2,2,2,3,3,3,3,4,4,2,2,2,2]
#dico_size = dico_size = {'small':range(1,2),
#           'intermediate':range(2,3),
#           'large':range(3,7)}
#word_from_MLP(li, 3, 2, dico_size) 
#[[[1, 2, 3], ['large', 'large', 'large'], 2],
# [[2, 3, 4], ['large', 'large', 'intermediate'], 2],
# [[3, 4, 2], ['large', 'intermediate', 'large'], 2]]
#word_from_MLP(li, 4, starting_hour, dico_size) 
#[[[1, 2, 3, 4], ['large', 'large', 'large', 'intermediate'], 2],
#[[2, 3, 4, 2], ['large', 'large', 'intermediate', 'large'], 2]]


def frequency_words(documents, path_save):
    li_words = []
    for h_day, li_li_tupleWordTime in documents.items():
        li_words.extend([x for x in li_li_tupleWordTime])
    #print(len(li_words), len(set(li_words)))
    c_words = Counter(li_words)
    ### frequence analysis
    #create a list of all words
    li_words = []
    for h_day, li_li_tupleWordTime in documents.items():
        li_words.extend([x for x in li_li_tupleWordTime])
    #print(len(li_words), len(set(li_words)))

    #compute frequencies
    c_words = Counter(li_words)
    df_word_frequence = pd.DataFrame.from_dict({'word':list(c_words.keys()),'frequence':list(c_words.values())})
    df_word_frequence = df_word_frequence.sort_values('frequence',ascending=False)
    df_word_frequence.to_csv(os.path.join(path_save,'word_Frequence.csv'),index=False,sep=';')
    #display(df_word_frequence.head(3))
    #display(df_word_frequence.tail(3))

    #simple barplot (sorted with x values)
    d = {k:v for k,v in c_words.items() if v>5}
    d = sorted(d.items(), key=operator.itemgetter(1))
    x = [i[0] for i in d]
    y = [i[1] for i in d]
    fig = plt.figure(figsize=(30,7))
    ax = plt.subplot(111)
    width = 0.8
    ax.bar(range(len(x)), y, width=width)
    plt.title('most frequent words')
    ax.set_xticks(np.arange(len(x)) + width/2)
    ax.set_xticklabels(x, rotation=90,size=6);
    plt.savefig(os.path.join(path_save,'most_frequent_word.png'),dpi=300,format='png',bbox_inches='tight')
    plt.clf()
    
    #histogram
    plt.hist(df_word_frequence[df_word_frequence['frequence']<=100]['frequence'],bins='auto') #into 15 equal parts 
    #return: [0]: vector of length bins with #elements in each bins
    #and [1]: when the vectors starts (for plot)
    plt.xlabel('frequencies')
    plt.ylabel('number of words')    
    #--> choose the nbr_times parameter: number of times a words need to appear at least this amount of time, in the overall set of 
    #documents to be taken into account
    plt.savefig(os.path.join(path_save,'histogram_of_word_frequencies.png'),dpi=300,format='png',bbox_inches='tight')
    plt.clf()
    return c_words


def topic_modelling_on_MLP(df_ts, type_, length_words, morethan4_in1, type_hybrid, dico_size, comment, nbr_times, li_LB, li_LSL, 
                           config, Sess2keep, starting_hour, max_topic=15, sec_threshold=60*5, all_=False):           
    #initialise parameters
    path_extracted_data = config.path_extracted_data
    path_initial_data = config.path_initial_data
    id_run = config.id_run

    ### unique naming
    title_ = type_hybrid+'---'+type_+'---'+comment+'---sess_'+'_'.join([str(x) for x in Sess2keep]) 
    #'_LB', '_all','_LSL' '_LB_newword_def' '_all_newword_def'
    print('This topic modeling will be save under the name: '+title_)
    #create a director if not existing
    path_save = os.path.join(path_extracted_data,'visual','TM', title_)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        
    ### adapt to the need (keep columns of one species only if wanted and rows with session that we want)
    print(df_ts.shape)
    if 'LB' in title_:
        df_ts = df_ts.filter([x for x in df_ts.columns if (not x.startswith('hen_')) | (x in li_LB)],axis=1)
    if 'LSL' in title_:
        df_ts = df_ts.filter([x for x in df_ts.columns if (not x.startswith('hen_')) | (x in li_LSL)],axis=1)
    print(df_ts.shape)
    df_ts['sessionID'] = df_ts['session'].map(lambda x: int(x[:-1]))
    #df_ts['sessionID'].value_counts()
    df_ts = df_ts[df_ts['sessionID'].isin(Sess2keep)]
    li_hen = [i for i in df_ts.columns if i.startswith('hen_')]
    print(df_ts.shape)
    
    ### documents:  set of words of daily hen time series
    documents = {}
    for day, df_ in df_ts.groupby(['day']):
        df_ = df_.fillna(' ')
        for h in li_hen:
            li = df_[h].tolist()
            #remove if nan in ts or if only one zone (we separate for efficiency)
            if li[0]!=' ':
                if len(set(li))>1:
                    if type_=='varyinglength':
                        documents[h+'/-/'+str(day).split(' ')[0]] = word_from_MLP_end_and_begin_atlongDuration(li, sec_threshold, 
                                                                                                               starting_hour,
                                                                                                               dico_size, config,
                                                                                                               morethan4_in1=morethan4_in1)
                    if type_ =='fixedLength':
                        documents[h+'/-/'+str(day).split(' ')[0]] = word_from_MLP(li, length_words, starting_hour, dico_size)
    #small visual
    #h = 'hen_45'
    #day = '2016-10-28'
    #display(documents[h+'/-/'+day])
    #plt.plot(df_ts[df_ts['day']==dt.datetime.strptime(day, '%Y-%m-%d')][h].tolist());
    #plt.title(h+'   '+day);

    ### compute frequencies of each word and clean the documents
    c_words = frequency_words(documents, path_save)
    cleaned_documents, tokens_removed = MLPsWord4lda(documents, c_words, nbr_times)
    #len(tokens_removed)

    ### dictionary & bag of word corpus
    #dictionary:  mapping from word IDs to words
    #keep track of the order of the li_documents regarding the ts ID
    dico_tsID_listID = dict(zip(list(cleaned_documents.keys()),range(len(cleaned_documents.keys()))))
    dico_listID_tsID = {v:k for k,v in dico_tsID_listID.items()}
    li_documents = [cleaned_documents[dico_listID_tsID[listID]] for listID in range(len(dico_listID_tsID))]
    print('We have %d documents (i.e. daily hens time series). The first one has %d words (once cleaned)'%(len(li_documents), 
                                                                                                           len(li_documents[0])))
    dictionary = corpora.Dictionary(li_documents)
    print('There is %d words in your dictionary'%len(dictionary))

    #corpus: list with a bag of words (sparse document vectors: list of tuples(word_id, appearance)) for each documents
    corpus = [dictionary.doc2bow(text) for text in li_documents]
    #print(len(corpus),corpus[10])

    #save
    pickle.dump(corpus, open(os.path.join(path_save,'corpus.pkl'), 'wb'))
    dictionary.save(os.path.join(path_save,'dictionary.gensim'))

    ### topic modelling
    for nbr_topics in range(2,max_topic): 

        #create a director if not existing
        path_save = os.path.join(path_extracted_data,'visual','TM', title_, str(nbr_topics))
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        #train lda
        lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, 
                                       num_topics=nbr_topics, 
                                       passes=30, 
                                       chunksize=50, 
                                       random_state=100,
                                       update_every=5, 
                                       alpha='auto', 
                                       per_word_topics=False)
        lda.save(os.path.join(path_save,'model'+str(nbr_topics)+'.gensim'))

        #show the latent topics
        #for topicsID_topicsWordDistribution in lda.print_topics():
        #    print('\n----',topicsID_topicsWordDistribution[0])
        #    print(topicsID_topicsWordDistribution[1])

        #summarize the results
        df_topics = pd.DataFrame(list(documents.items()),columns=['documentID','li_words'])
        df_topics['li_words_cleaned'] = df_topics['documentID'].map(lambda x: cleaned_documents[x])
        df_topics['corpus'] = df_topics['li_words_cleaned'].map(lambda x: dictionary.doc2bow(x))
        df_topics['lda_corpus'] = df_topics['corpus'].map(lambda x: lda[x])

        #add info on hen
        df_topics['HenID'] = df_topics['documentID'].map(lambda x: x.split('/-/')[0])
        df_topics['day'] = df_topics['documentID'].map(lambda x: dt.datetime.strptime(x.split('/-/')[1], '%Y-%m-%d'))

        #add info of topic 
        df_topics['topic_info'] = df_topics['lda_corpus'].map(lambda x: sorted(x,key=itemgetter(1))[-1])
        df_topics['topic'] = df_topics['topic_info'].map(lambda x: x[0])
        df_topics['topic_proba'] = df_topics['topic_info'].map(lambda x: x[1])
        for t in range(nbr_topics):
            df_topics['topic_'+str(t)+'_proba'] = df_topics['lda_corpus'].map(lambda x: max([i[1] for i in x if i[0]==t]+[0]))

        #save
        df_topics.to_csv(os.path.join(path_save,'df_topics'+str(nbr_topics)+'.csv'), index=False,sep=';')
        #print(df_topics.shape)
        df_topics.head(3)
        #print(df_topics['topic'].value_counts())
        if len(df_topics['topic'].unique())!=nbr_topics:
            print('WARNING: to many topics, there is only %d instead of the %d wanted, we will stop now'%(len(df_topics['topic'].unique()), nbr_topics))
            return 
        ### topics repartition across hens
        df_plot = df_topics.groupby(['HenID','topic']).size().reset_index().pivot(columns='topic', index='HenID', values=0)
        df_plot = df_plot.fillna(0)
        #display(df_plot.head(3))
        df_plot_normalized = df_plot.div(df_plot.sum(axis=1)*0.01, axis=0).sort_values([0]) #sort according to subject 0
        #display(df_plot_normalized.head(3))
        li_color = sns.color_palette("RdBu", nbr_topics)
        df_plot_normalized.plot(x=df_plot_normalized.index, kind='bar', stacked=True, figsize=(30,7), 
                                legend=True, color=li_color).legend(bbox_to_anchor=(1.2, 0.5));
        plt.savefig(os.path.join(path_save,'topic_repartition_across_hen_'+str(nbr_topics)+'.png'), 
                    dpi=300, format='png', bbox_inches='tight')
        plt.clf()

        ### topic mixture across documents    
        if all_:
            for t in range(nbr_topics):
                df_plot_normalized = df_topics[['topic_'+str(i)+'_proba' for i in range(nbr_topics)]].sort_values(['topic_'+str(t)+'_proba'])
                df_plot_normalized.plot(x=df_plot_normalized.index, kind='bar', stacked=True, figsize=(40,10), 
                                        legend=True).legend(bbox_to_anchor=(1.2, 0.5));
                plt.savefig(os.path.join(path_save, 'topic_mixture_across_document_'+str(t)+'.png'),dpi=300,format='png',bbox_inches='tight')
                plt.clf()

        ### topics appearance across days
        df_plot = df_topics.sort_values('day', ascending=True).copy()
        #dico_topicID_color = {0:(0.979891, 0.90894778, 0.84827858), 1:'blue', 2:'yellow', 3:'green', 4:'grey', 5:'lime', 6:'orange'}
        li_color = sns.color_palette("rocket_r", nbr_topics)
        c = 2
        if nbr_topics%3==0:
            c = 3
        l = math.ceil(nbr_topics/c)
        fig = plt.figure(figsize=(c*5, l*5))
        x = df_plot['day'].unique()
        for i, (t, df_) in enumerate(df_plot.groupby(['topic'])):
            ax = plt.subplot(l,c,i+1)
            df_ = df_.groupby(['day'])['documentID'].count().reset_index()
            x = df_['day'].tolist()
            y = df_['documentID'].tolist()
            #print(len(x))
            plt.plot(x,y,color=li_color[i])
            plt.ylim(0,max(y)+max(int(max(y)*0.1),1))
            for label in ax.get_xticklabels():
                label.set_rotation(25) 
            plt.title('topic   '+str(t))
        plt.savefig(os.path.join(path_save,'topic_appearance_across_days_.png'),dpi=300,format='png',bbox_inches='tight')
        plt.clf()
        #display(df_.head(3))    

        ### hens main topics across days
        if all_:
            df_plot = df_topics.sort_values('day', ascending=True).copy()
            #x = list(set(df_plot['day'].tolist()))
            for t in range(nbr_topics):
                fig = plt.figure(figsize=(20,6))
                for i, (h, df_) in enumerate(df_plot.groupby(['HenID'])):
                    x = df_['day'].tolist()
                    y = df_['topic_'+str(t)+'_proba'].tolist()
                    plt.plot(x,y)
                    fig = plt.figure(figsize=(20,6))
                    if i==5:
                        sys.exit()
                plt.title('topic   '+str(t))

        ### main topics proba
        plt.hist(df_topics['topic_proba'].dropna(), bins='auto')
        plt.xlabel('main topic predominance')
        plt.ylabel('number of documents (daily hen ts)');
        plt.savefig(os.path.join(path_save,'main_topic_proba.png'),dpi=300,format='png',bbox_inches='tight')
        plt.clf()

        ### plot visual time series into cluster-folders
        for i in range(0,nbr_topics):
            path_ts = os.path.join(path_save, 'timeseries_plot_cluster', str(i))
            if not os.path.exists(path_ts):
                os.makedirs(path_ts)

        path_ = r'D:\vm_exchange\AVIFORUM\data\extracted_info_mobility_VF\visual\TimeSeriesPlot\time_series_plot'

        for i in range(df_topics.shape[0]):
            HenID = df_topics.iloc[i]['HenID']
            day = df_topics.iloc[i]['day']
            topic = df_topics.iloc[i]['topic']
            image_name = 'VF_'+str(day).split(' ')[0]+'_'+HenID+'.png'
            shutil.copy(os.path.join(path_, image_name), 
                        os.path.join(path_save, 'timeseries_plot_cluster', str(topic), image_name))

        #LDA nice visual
        lda_model = gensim.models.ldamodel.LdaModel.load(os.path.join(path_save, 'model'+str(nbr_topics)+'.gensim'))
        lda_display = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
        pyLDAvis.display(lda_display)
        pyLDAvis.save_html(lda_display, os.path.join(path_save, str(nbr_topics)+'_lda.html'))

        
        
##########################################################################################################################################
################################################################# PCA ####################################################################
##########################################################################################################################################

def pca_fct(df, caracteristics):
    
    '''We replace the missing value by the mean of the column. PCA on continuous caracteristics'''
    
    #take only the target and caracteristics columns of the dataframe
    df = df.filter(caracteristics,axis=1).reset_index()
    #replace missing value by the mean of the column
    df = df.fillna(df.mean())
    #normalize the data
    df = pd.DataFrame(scale(df), columns=df.columns)
    #sort the dataframe with respect to the target, so that with the color it will be easier to verify
    X = df[caracteristics]
    X = X.as_matrix()
    
    #fix a random seed in order to have always the same plots 
    np.random.seed(5)
    #plot
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    #ax = fig.add_subplot(111, projection='3d')
    plt.gca().patch.set_facecolor('black')

    pca = decomposition.PCA(n_components=len(caracteristics)-1)
    print (pca)
    pca.fit(X)
    X = pca.transform(X)
    
    #3Dplot
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=50)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    # Make an axis for the colorbar on the right side
    #cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    #im = ax.imshow(color, vmin=0, vmax=2)
    #fig.colorbar(im, cax=cax)
    plt.show()
    
    #Cumulative Variance explained
    var = pca.explained_variance_ratio_
    var1 = np.cumsum(np.round(var, decimals=4)*100)
    plt.plot(var1)
    plt.title('Cumulative Variance explained')
    plt.show()

    return pd.DataFrame(X)        
        
    
##########################################################################################################################################
############################################################ Clustering & PCA ############################################################
##########################################################################################################################################
        
def kmeans_clustering(df, range_n_clusters, drop_col_list=[]):
    '''The Nan entry will be converted to the mean of the columns, so that they do not really influence the classification.
    The kmeans packages allows you to do clustering on continuous variables, for categorical variables use kmodes.
    This function output firstly a list of the class label for each node, secondly some more information (centers)'''
    
    #drop some values if asked (often for the id)
    if len(drop_col_list)>0:
        df = df.drop(drop_col_list,axis=1)    
    #replace Na values by the mean, so that when we sum columns we dont loose information. TO BE CHANGED WHEN NEEDED
    df.fillna(df.mean(),inplace=True)
    
    #choosing the right number of cluster
    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(df)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(df, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)   
    k = input('Please let me know the numbers of clusters you want to search for ')
                    
    #apply kmeans
    print(df.shape)
    km = KMeans(n_clusters=int(k),random_state=3).fit(df)
    result = km.labels_
    return (result, km.cluster_centers_)


#function performing a kmodes clustering
def kmodes_clustering(df, nbr_cluster=3, op={}, drop_col_list=[]):
    '''The kmodes packages allows you to do clustering on categorical variables, otherwise use kmeans.
    This function output first a list of the class label for each node, then some more information (centroids)'''
    
    #drop some values if asked
    if len(drop_col_list)>0:
        df = df.drop(drop_col_list,axis=1)    
            
    #change the dataframe if asked     
    if len(op)>0:
        #create the good dataframe
        total = []
        df2 = pd.DataFrame(index=range(0,df.shape[0]), columns=op.keys())
        for key, col_list in collections.OrderedDict(sorted(op.items(), key=lambda t: t[0])).items():
        #for key, col_list in op.items():
            total.extend(col_list)
            df2[key] = df[col_list].sum(axis=1) / len(col_list)
        if len(total) != df.shape[1]:
            print('We will drop out the columns you havent mentioned, that is %s' %str(set(df.columns)-set(total)))
        df = df2
        
    #apply kmodes
    #print(df.columns)
    kmodes_cao = kmodes.kmodes.KModes(n_clusters=nbr_cluster, init='Cao', verbose=2)
    result = kmodes_cao.fit_predict(df, categorical=[1])
    return (result, kmodes_cao.cluster_centroids_)

from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition


#PCA (only with continous caracteristics)
def pca_fct(df,caracteristics,target=None,rs=0):
    
    '''give a target if you want the color of the dots to represent a certain nuemrcial values, it will not be
    taken into account in the PCA. The plot will de bonne with the three comp of a PCA with 3princ comp. the output
    will be based on a PCA having #caracteristics-1 possible components
    
    this function would be appropriate to check if a feature (e.g. a health indicator) could be explained by a big set of 
    varibales (PCA on this set, then plot and color with the features)
    
    the fct also fill nan with the mean of the column, and scale the dataframe beforehand'''
    
    #filter the target and caracteristics columns of the dataframe
    all_var = caracteristics + [target]
    df = df.filter(all_var,axis=1).reset_index()
    if target!=None:
    #remove rows that have a missing values in the target entry
        print(df.shape[0])
        df.dropna(subset=[target],inplace=True)
        print(df.shape[0])
    #replace missing value by the mean of the column
    df = df.fillna(df.mean())
    #normalize the data
    df = pd.DataFrame(scale(df), columns=df.columns)

    #sort the dataframe with respect to the target, so that with the color it will be easier to verify
    if target!=None:
        df = df.sort_values(target)
        #computing the target and caracteristics numpy array
        y = df[[target]]
    X = df[caracteristics].values
    
    #in order to have always the same plots 
    np.random.seed(rs)
    #plot
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    #ax = fig.add_subplot(111, projection='3d')
    plt.gca().patch.set_facecolor('black')
    #ax.w_xaxis.set_pane_color((0.0, 0.8, 0.5, 0.0))
    #ax.w_yaxis.set_pane_color((0.5, 0.8, 0.5, 0.0))
    #ax.w_zaxis.set_pane_color((1.0, 0.8, 0.5, 0.0))
    
    pca1 = decomposition.PCA(n_components=len(caracteristics)-1)
    pca1.fit_transform(X)
    var = pca1.explained_variance_ratio_
    #print(var)
    #ax = Axes3D(fig, rect=[0, 0, .95, 1])
    #plt.cla()
    
    
    ############### Visual on another pca, with only 3 components
    pca = decomposition.PCA(n_components=3)
    #print(pca)
    pca.fit(X)
    X = pca.transform(X)
    print ('The importance of the component are the following:')
    print (pca.explained_variance_ratio_)
    print('and the sum is:%f' %sum(pca.explained_variance_ratio_))
    # Dump components relations with features:
    #print (pd.DataFrame(pca.components_,columns=df[caracteristics].columns,index = ['PC-1','PC-2','PC-3']))
    
    if target!=None: 
        #color = [(i[0]/155.,5/(50-i[0]),15/i[0]) for i in y.values.tolist()]
        color = [i[0]/5. for i in y.values.tolist()]
        #color = [0.242484 for i in y.values.tolist()]
        #color[2] = 0.3
        #color[1] = 0.3
        #color[0] = 0.3
        print(plt.cm.Spectral)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color,s=50)
        col = pd.Series(color).reset_index()
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        plt.show()

        return pd.concat([df,col],axis=1),var

    else:
        #print(plt.cm.Spectral)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=50)
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])

        # Make an axis for the colorbar on the right side
        #cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        #im = ax.imshow(color, vmin=0, vmax=2)
        #fig.colorbar(im, cax=cax)
        plt.show();

        return pd.concat([df],axis=1),var

#e.g. of usage
#target can typically be the output of a health indicator 
#df_pca, var = pca_fct(df_,li_all)

#Cumulative Variance explains
#var1 = np.cumsum(np.round(var, decimals=4)*100)
#plt.plot(var1)
#plt.show()

##########################################################################################################################################
######################################################### variable computation ###########################################################
##########################################################################################################################################

#taking from: https://en.wikipedia.org/wiki/Sample_entropy
def sampen(L, r=None, m=2):
    '''Usually m=2 and r=0.2*std'''
    N = len(L)
    B = 0.0
    A = 0.0
    
    if r==None:
        r = 0.2*np.std(L)
    
    # Split time series and save all templates of length m
    xmi = np.array([L[i:i+m] for i in range(N-m)])
    xmj = np.array([L[i:i+m] for i in range(N-m+1)])
       
    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii-xmj).max(axis=1) <= r)-1 for xmii in xmi])
            
    # Similar for computing A
    m += 1
    xm = np.array([L[i:i+m] for i in range(N-m+1)])
        
    A = np.sum([np.sum(np.abs(xmi-xm).max(axis=1) <= r)-1 for xmi in xm])

    # Return SampEn
    return -np.log(A/B)

#MSSD (Mean of the squared successive differences)
#It is calculated by taking the sum of the differences between consecutive observations squared, then taking the mean of that 
#sum and dividing by two
def mssd(li, maxnbr_nan=None):
    li_diff = [(li[i+1]-li[i])**2 for i in range(0,len(li)-1) if (math.isnan(li[i+1])==False)&(math.isnan(li[i])==False)]
    #return NA if to many nan
    if maxnbr_nan!=None:
        if (len(li)-len(li_diff))>=maxnbr_nan:
            return(np.nan)
    return(np.mean(li_diff)/2)


#computing chi2-distance
def chi2_distance(l1,l2,remove_warning=False):
    '''compute the following distance: d(x,y) = sum( (xi-yi)^2 / (xi+yi) ) / 2
    allow to be flexible regarding chi2distance for exemple for lda mixture of topics'''
    if len(l1)!=len(l2):
        print('your two vectors must have same length')
        sys.exit()
    if (abs(sum(l1)-1)>0.01) | (abs(sum(l2)-1)>0.01):
        if not remove_warning:
            print('WARNING: your two vectors must be normalized (sumed to one) for now their sum are:',abs(sum(l1)-1),'and', abs(sum(l2)-1))
            print('We will normalise them so that their sum equals one')
        l1 = [i/sum(l1) for i in l1]
        l2 = [i/sum(l2) for i in l2]
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

#same but nicer:
def list_of_zones(li):
    return [x[0] for x in itertools.groupby(li)]
#small example
#li = [1,1,1,1,2,2,2,3,3,3,4,4,2,2,2,2,4,1,2]
#list_of_zones(li) #[1, 2, 3, 4, 2, 4, 1, 2]

def list_Z4(li):
    return [1 if '4' in str(x) else 0 for x in li]
#small example
#li = [1,1,1,1,2,2,2,3,3,3,4,4,2,2,2,2,4,1,2]
#list_of_Z4(li) #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]

#accounting for all the transitional zones
def dict_of_zones_appearances_with_transitionalZones(li):
    '''must be numerical'''
    #remove the effect of duration
    li = [x[0] for x in itertools.groupby(li)]
    #add transitional zones (take only inbetween min and max zone) : note that the order does not mean anything
    li_trans = [list(range(min(li[i],li[i+1]),max(li[i],li[i+1])))[1:] for i in range(0,len(li)-1)]
    li_trans = [x for i in li_trans for x in i]
    #print(li_trans)
    li.extend(li_trans)
    return dict(Counter(li))   
#small example
#li = [1,1,1,1,2,2,2,3,3,4,4,2,2,4,1,2] 
#dict_of_zones_appearances_with_transitionalZones(li) #li_trans: [3, 3, 2, 3], output: {1: 2, 2: 4, 3: 4, 4: 2}


def empirical_probabilites_of_goingup(li):
    #turn into a sequence (i.e. excluding duration of stay in a zone)
    li = [x[0] for x in itertools.groupby(li)]
    dico_z_empproba = {}
    #look only at inbetween zones
    for z in [2,3,4]:
        #True: means going up, False: mean going down
        #we count if the zone appears in between two numbers but is not the last number (i.e. as in that case its going to z, not
        #going from z!)
        l = [(li[i+1]-li[i])>0 for i in range(0,len(li)-1) if ((z<=max(li[i],li[i+1]))&(z>=min(li[i],li[i+1]))&(z!=li[i+1]))]
        #print(l)
        dico_z_empproba[z] = np.nan
        if len(l)>0:
            dico_z_empproba[str(z)+'_Zone'] = sum(l)/len(l)*100
    return dico_z_empproba
#small example
#li = [1,1,1,1,2,2,2,3,3,4,4,2,2,4,1,2]
#empirical_probabilites_of_goingup(li) #{2: 0.6666666666666666, 3: 0.5, 4: 0.0}


def nbr_bouts_per_zone(li):
    '''function to compute number of time a hen went into a zone (i.e. if already in a zone, then it takes it into account, althoguht it is not an actual transition'''
    v = [i[0] for i in itertools.groupby(li)]
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
    '''function to find zone(s) where the hen staid the longest consecutively'''
    v = [(x[0], len(list(x[1]))) for x in itertools.groupby(li)]
    v = sorted(v, key = lambda i: i[1])
    m = max([i[1] for i in v])
    v = [i[0] for i in v if i[1]==m]   
    if len(set(v))==1:
        return(v[0])
    else:
        print('several max-duration-zone')
        return(v)
    
def duration_Z5(li):
    '''function to find time spend in Z5'''
    return sum(['5' in str(x) for x in li])

############statistics on duration
#where for li = [1,1,1,1,2,2,2,3,3,3,4,4,2,2,2,2] we get if nbr_sec=3: v= [12, 9, 9, 6, 12] which we then aggreagate (min, max, avg,...)
def list_of_durations(li, nbr_sec):
    return [len(list(x[1]))*nbr_sec for x in itertools.groupby(li)]

def zone_duration(li,nbr_sec):
    li_dur = list_of_durations(li,1)
    return [[li[sum(li_dur[0:i])] for i in range(len(li_dur))],li_dur]
#li = [1,1,1,1,4,2,2,2,3,3,3,3,4,4,2,2,2,2]
#zone_duration(li,1) #[[1, 4, 2, 3, 4, 2], [4, 1, 3, 4, 2, 4]]

def dico_rule_output(i,dico):
    for k,v in dico.items():
        if i in v:
            return k      

def max_duration(li, nbr_sec):
    '''function to compute the maximum of the durations in minutes in any zone'''
    v = [(len(list(x[1]))*nbr_sec)/60 for x in itertools.groupby(li)]
    return max(v)

def stats_duration(li, nbr_sec):
    '''function to compute the kurtosis and skewnedd of the list of duration in any zone'''
    v = [(len(list(x[1]))*nbr_sec)/60 for x in itertools.groupby(li)]
    return kurtosis(v), skew(v)

def var_duration(li, nbr_sec):
    '''function to compute the variation of the durations in minutes in any zone'''
    #list of durations
    v = [(len(list(x[1]))*nbr_sec)/60 for x in itertools.groupby(li)]
    return np.var(v) 

def min_duration(li, nbr_sec):
    '''function to compute the min of the durations in minutes in any zone'''
    #list of durations
    v = [(len(list(x[1]))*nbr_sec)/60 for x in itertools.groupby(li)]
    return min(v) 

def median_duration(li, nbr_sec):
    '''function to compute the median of the durations in minutes in any zone'''
    v = [(len(list(x[1]))*nbr_sec)/60 for x in itertools.groupby(li)]
    return np.median(v)

def average_duration(li, nbr_sec):
    '''function to compute the average of the durations in minutes in any zone'''
    v = [(len(list(x[1]))*nbr_sec)/60 for x in itertools.groupby(li)]
    return np.mean(v)

def dico_duration_stats(li, nbr_sec):
    '''function to compute some stats on the list of duration (in seconds) in any zone'''
    v = [(len(list(x[1]))*nbr_sec) for x in itertools.groupby(li)]
    dico_dur_stats =  {'mean_duration':np.mean(v), 'median_duration':np.median(v), 'max_duration':max(v),
                       'min_duration':min(v), 'variance_duration':np.var(v), 'percentile_5':np.percentile(v,5),
                       'percentile_15_duration':np.percentile(v,15,interpolation='lower'),
                       'percentile_85_duration':np.percentile(v,85,interpolation='lower'),
                       'percentile_95_duration':np.percentile(v,95,interpolation='lower'),
                       'median_abs_deviation_duration':median_abs_deviation(v),
                       'kurtosis_duration':kurtosis(v), #todo: think of fisher=True, bias=True
                       'skew_duration':skew(v)} 
    return dico_dur_stats


#li = [1,1,1,1,1,2,2,2,2,3,3,3,4,4,4,2,2,2,2,1,1,1,1]
#[list(x[1]) for x in itertools.groupby(li)]
#[[1, 1, 1, 1, 1],
# [2, 2, 2, 2],
# [3, 3, 3],
# [4, 4, 4],
# [2, 2, 2, 2],
# [1, 1, 1, 1]]


############################ Food related behavior
#Food related behavior: (zone 3 and zone 5?): (DZ3FR/DFR&inside - DZ3FNR/DFNR&inside) /(DZ3FR/DFR&inside + DZ3FNR/DFNR&inside). birds that stays in the top tier to always for instance should have ~0 issue: two zones
def food_related_behavior(li_Z, config, fake_sec=0):
    '''fake_sec allows to repeatability of this behavior with ne having random/nonfood related timing. We will be moving the list of second 0/1:feeding/not feeding by fake_sec. fake_sec should be negative if we want to move the feeding time to earlier timestamps. FOr example, fake_sec of 60*2 means that we will pretend the food is delivered 2mn later so we will add 0 at begining and cut the last ones'''
    #initialise parameters
    li_FR = config.li_FR
    li_FNR = config.li_FNR
    
    #add fake_sec
    if fake_sec>0:
        li_FR =  list([0] * fake_sec)+li_FR[:-fake_sec]  
        li_FNR =  list([0] * fake_sec)+li_FNR[:-fake_sec]  
    if fake_sec<0:
        li_FR =  li_FR[abs(fake_sec):] + list([0] * fake_sec)
        li_FNR =  li_FNR[abs(fake_sec):] + list([0] * fake_sec)
        
    #from 1_zone to 1
    li_Z = [int(i.split('_')[0]) for i in li_Z]
      
    li_foodrunning_birdinside = [int(f==1)*int(z>1) for f, z in zip(li_FR, li_Z)]
    #print(li_foodrunning_birdinside)
    
    li_foodNOTrunning_birdinside = [int(fnr==1)*int(z>1) for fnr, z in zip(li_FNR, li_Z)]
    #print(li_foodNOTrunning_birdinside)

    li_foodruning_birdZ3Z5 = [int(f==1)*int(z in [3,5]) for f, z in zip(li_FR, li_Z)]
    #print(li_foodruning_birdZ3Z5)

    li_foodNOTruning_birdZ3Z5 = [int(fnr==1)*int(z in [3,5]) for fnr, z in zip(li_FNR, li_Z)]
    #print(li_foodNOTruning_birdZ3Z5)
    
    if sum(li_foodrunning_birdinside)==0:
        return np.nan
    
    if sum(li_foodNOTrunning_birdinside)==0:
        return -9999
    n = sum(li_foodruning_birdZ3Z5)/sum(li_foodrunning_birdinside) - sum(li_foodNOTruning_birdZ3Z5)/sum(li_foodNOTrunning_birdinside)
    p = sum(li_foodruning_birdZ3Z5)/sum(li_foodrunning_birdinside) + sum(li_foodNOTruning_birdZ3Z5)/sum(li_foodNOTrunning_birdinside)
    #print(n, p)
    
    #if never went in the Z3&Z5 then return 0 then division by 0 issue --> return 0 i.e. neutral
    if p==0:
        return(0)
    return n/p
#li_ZS = [1,1,1,1,2,2,2,3,3,3,1,4,2,2,2,2,5,5,5,4,4,4]
#li_FR = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0]
#print(food_related_behavior(li_FR=li_FR, li_Z=li_ZS, config=config))-0.11428571428571432 0.6857142857142857 --> -0.17
#li_ZS = [1,1,1,1,2,2,2,3,3,3,3,3,3,3,2,2,2,2,2,4,4,4]
#li_FR = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0]
#print(food_related_behavior(li_FR=li_FR, li_Z=li_ZS, config=config))0.65 0.85 --> 0.7647058823529412

############################ nestboxes related behavior
def nestboxes_related_behavior(li, config, li_LT=[], li_HT=[]):
    '''function that compute a proxi for nestbox zone usgae: morning vs afternoon. values> means that you have an afternoon 
    usage of the nestbox (maybe hidding), while <0 you have a morning usage of nestboxes (maybe laying)'''
    
    #initialise parameters
    li_LT = config.li_LT
    li_HT = config.li_HT
    
    #from 1_zone to 1
    li = [int(i.split('_')[0]) for i in li]
    
    li_layingtime_birdinZ4 = [int(lt==1)*int(z==4) for lt, z in zip(li_LT, li)]    
    li_hidingtime_birdinZ4 = [int(ht==1)*int(z==4) for ht, z in zip(li_HT, li)]    
    p = sum(li_layingtime_birdinZ4)/sum(li_LT)+sum(li_hidingtime_birdinZ4)/sum(li_HT)
    n = sum(li_layingtime_birdinZ4)/sum(li_LT)-sum(li_hidingtime_birdinZ4)/sum(li_HT)
    #if never went in the nestbox then return 0 then division by 0 issue --> return 0 i.e. neutral nor afternoon nor morning:
    #i.e. afternon and morning have same behavior
    if p==0:
        return(0)
    return n/p
#small verifications
#li_ZS = [1,1,4,4,4,1,1,2,2,2,3,3,3,1,4,2,2,2,2,5,5,5,4,4,4]
#li_LT = [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#li_HT = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
#print(nestboxes_related_behavior(li=li_ZS, config=config, li_LT=li_LT, li_HT=li_HT)) #(3/9 - 4/12)/(3/9 + 4/12) = 0)
#li_ZS = [1,1,1,1,1,1,1,2,2,2,3,3,3,1,4,2,2,2,2,5,5,5,4,4,4]
#li_LT = [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#li_HT = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
#print(nestboxes_related_behavior(li=li_ZS, config=config, li_LT=li_LT, li_HT=li_HT)) #(0/9 - 4/12)/(0/9 + 4/12) = -1)
#li_ZS = [1,1,1,1,1,1,1,2,2,2,3,3,3,1,1,2,2,2,2,5,5,5,1,1,1]
#li_LT = [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#li_HT = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
#print(nestboxes_related_behavior(li=li_ZS, config=config, li_LT=li_LT, li_HT=li_HT)) #(0/9 - 0/12)/(0/9 + 0/12) = 0)
#li_ZS = [4,4,4,4,4,1,1,2,2,2,3,3,3,1,4,2,2,2,2,5,5,5,4,4,4]
#li_LT = [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#li_HT = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
#print(nestboxes_related_behavior(li=li_ZS, config=config, li_LT=li_LT, li_HT=li_HT)) #(5/9 - 4/12)/(5/9 + 4/12) = 0.25)

def mid_cum_Z4_sec(li):
    li = list(np.cumsum([int('4' in str(x)) for x in li]))
    m = int(max(li)/2) #round down 
    if m>0:
        ind_ = li.index(m) + 1 #+1 for index 0
        return(ind_)
    #if never went in nestboxes then put np.nan
    return np.nan
#small example
#li = [1,1,4,4,4,1,1,2,2,2,3,3,3,1,4,2,2,2,2,5,5,5,4,4,4]
#mid_cum_Z4(li) #5

############### missing zones (flying)
def li_missingZone_mvtPerc_DU(li,nbr_sec):
    li_z_d_pzd = list_tuple_zone_dur_previousZoneDiff(li, nbr_sec)
    missingdown_perc = 0
    missingup_perc = 0
    if len(li_z_d_pzd)>1:
        missingdown_perc = sum([1 for z,d,pzd in li_z_d_pzd[1:] if pzd>1])/(len(li_z_d_pzd)-1)*100
        missingup_perc = sum([1 for z,d,pzd in li_z_d_pzd[1:] if pzd<-1])/(len(li_z_d_pzd)-1)*100
    return missingdown_perc, missingup_perc, len(li_z_d_pzd)
#small examples
#li = [1,1,1,1,1,2,2,2,2,3,3,3,2,2,2,2,4,4,4,2,2,2,2,1,1,1,1]
#print(li_missingZone_mvtPerc_DU(li,1)) 
#(16.666666666666664, 16.666666666666664, 7)
#li = [1,2,3,2,4,2,1]
#print(li_missingZone_mvtPerc_DU(li,1)) #, 1/6*100=16,66 
#(16.666666666666664, 16.666666666666664, 7)
#li = [1,2,3,2,4,3,2,1] #lenght = 7, 1/7*100=14
#print(li_missingZone_mvtPerc_DU(li,1)) 
#(0.0, 14.285714285714285, 8)
#li = [1,2,3,2,4,1] #lenght = 7, 1/5*100=40
#print(li_missingZone_mvtPerc_DU(li,1)) 
#(20.0, 20.0, 6)
#4 to 1 or 4 to 2 count the same, it count as one flying-mvt. The percenatge is taken over the entire nbr of transitions

############### chaotic transition
def list_tuple_zone_dur_previousZoneDiff(li, nbr_sec):
    li_dur = [len(list(x[1]))*nbr_sec for x in itertools.groupby(li)]
    li_zone = [list(x[1])[0] for x in itertools.groupby(li)]
    li_previous_zone = [None]+li_zone[:-1]
    li_previouszone_diff = [None]+[i - j for i, j in zip(li_previous_zone[1:], li_zone[1:])]
    return list(zip(li_zone, li_dur, li_previouszone_diff))

def li_event_chaoticmvt_z_d(li, nbr_sec, nbr_sec_chaoticmvt_notmiddle, dico_zone_order):
    '''gives more information on the zone and duration of the chaotic events'''
    dico_order_zone = {v:k for k,v in dico_zone_order.items()}
    li_z_d_pzd = list_tuple_zone_dur_previousZoneDiff(li, nbr_sec)
    li_info_z_d = [(dico_order_zone[li_z_d_pzd[i+1][0]],li_z_d_pzd[i+1][1]) if (li_z_d_pzd[i+0][1]>nbr_sec_chaoticmvt_notmiddle) \
                   & (li_z_d_pzd[i+2][1]>nbr_sec_chaoticmvt_notmiddle) \
                   & (li_z_d_pzd[i+1][2]==-li_z_d_pzd[i+2][2]) else (None,None) for i in range(0,len(li_z_d_pzd)-2)]
    return li_info_z_d
#small example
#li = [1,1,1,1,1,2,2,2,2,3,3,3,2,2,2,2,4,4,4,2,2,2,2,1,1,1,1]
#plt.plot(li)
#nbr_sec_chaoticmvt = 2
#print(list_tuple_zone_dur_previousZoneDiff(li, nbr_sec)) 
#[(1, 5, None), (2, 4, -1), (3, 3, -1), (2, 4, 1), (4, 3, -2), (2, 4, 2), (1, 4, 1)]
#print(li_event_chaoticmvt_z_d(li,1))
#[(3, 3), (None, None), (4, 3), (None, None)]
#another small examples:
#li = [1,1,1,1,1,2,2,2,2,3,3,2,2,2,2,4,4,4,4,2,2,2,2,1,1,1,1,4,1,4,1]
#li_event_chaoticmvt_z_d(li,1,0, dico_zone_order) 
#[(None, None),('4_Zone', 2), (None, None), ('5_Zone', 4), (None, None),(None, None), ('5_Zone', 1), ('2_Zone', 1), 
#('5_Zone', 1)]


def stats_chaoticmvt(li):
    li = [i for i in li if i[0]!=None]
    dico_z_lidur = {}
    for z,d in li:
        t = 'chaoticmvt_Middle'+str(z)
        if t not in dico_z_lidur:
            dico_z_lidur[t] = []
        dico_z_lidur[t].append(d)
    return dico_z_lidur
#small example
#x = df_daily['li_event_chaoticmvt_z_d'].iloc[i]
#stats_chaoticmvt(x)
#{'chaoticmvt_Middle3': [1721.0,  830.0,  2242.0,  1758.0,  79.0,...],
#'chaoticmvt_Middle4': [2934.0,1061.0,  17.0,  4118.0,  900.0,  2607.0,...]}

###############

def dico_zone_sortedduration(li, nbr_sec):
    '''function to find a list of duration per zone sorted from smaller to bigger'''
    v = [(x[0], len(list(x[1]))) for x in itertools.groupby(li)]
    v = sorted(v, key = lambda i: i[1]) #sort
    d = {}
    for i,j in v:
        if i not in d:
            d[i] = []
        d[i].append(j*nbr_sec)
    return d    
#small test
#t = [1,1,1,1,2,2,2,3,3,3,4,4,2,2,2,2,]
#max_duration(t), min_duration(t), median_duration(t), max_duration_zones(t), average_duration(t), dico_zone_sortedduration(t)
#--> several max-duration-zone
#12, 6, 9.0, [1, 2], 9.600000000000001, {1: [12], 2: [9, 12], 3: [9], 4: [6]}

def dico_zone_notsortedduration(li, nbr_sec):
    '''function to find a list of duration per zone sorted from smaller to bigger'''
    v = [(x[0], len(list(x[1]))) for x in itertools.groupby(li)]
    d = {}
    for i,j in v:
        if i not in d:
            d[i] = []
        d[i].append(j*nbr_sec)
    return d    
#small ex
#t = [1,1,1,1,2,2,2,3,3,3,4,4,2,2,2,2,1]
#dico_zone_sortedduration(t,1), dico_zone_notsortedduration(t,1), dico_zone_nbrminStartswith(t,1)
#({1: [1, 4], 2: [3, 4], 3: [3], 4: [2]},
# {1: [4, 1], 2: [3, 4], 3: [3], 4: [2]},
# {1: [1, 17], 2: [5, 13], 3: [8], 4: [11]})

def dico_zone_nbrminStartswith(li, nbr_sec):    
    v = [(x[0], len(list(x[1]))) for x in itertools.groupby(li)]
    v = [(v[i][0],sum([v[j][1] for j in range(i)])+1) for i in range(len(v))]
    d = {}
    for i,j in v:
        if i not in d:
            d[i] = []
        d[i].append(j*nbr_sec)
    return d   
#small exemple
#li = [1,1,1,2,2,2,3,3,4,4,2,2,1,1,1,4]
#dico_zone_nbrminStartswith(li, 1) #{1: [1, 13], 2: [4, 11], 3: [7], 4: [9, 16]}

def li_boots_starting_sec(li, nbr_sec):    
    v = [(x[0], len(list(x[1]))) for x in itertools.groupby(li)]
    v = [(v[i][0],sum([v[j][1] for j in range(i)])+1) for i in range(len(v))]
    return [k[1]*nbr_sec for k in v]
#small exemple
#li = [1,1,1,2,2,2,3,3,4,4,2,2,1,1,1,4]
#dico_boots_starting_sec(li, 1) #[1, 4, 7, 9, 11, 13, 16]


#not linked to the order, only the distribution (proba of a certain zone)
def old_DistributionEntropy(labels):
    '''compute the distribution entropy'''
    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts)
#small test
#note that entropy([4,4,0,0,0,0,0,0,0,0]) == entropy([4,4])
#Note: entropy([500,500]) == entropy([5,5])
#print(DistributionEntropy([1,1,1,1,1,1,1,1,0,2,3,4,5,6,7,8,9,9,9,9]), 
#      DistributionEntropy([0,2,3,4,5,6,7,8,9,9,9,9,1,1,1,1,1,1,1,1]),
#      DistributionEntropy([1,1,1,1,1,1,1,1]), 
#      DistributionEntropy([1,0,1,1,1,1,1,1,0]), 
#      DistributionEntropy([0,1,1,1,1,1,1,0,1]), 
#      DistributionEntropy([1,1,0,0,1,1,1,1,1]),
#      DistributionEntropy(['1','1','8','9','9'])) #categorical values oke
#1.8866967846580784 1.8866967846580784 0.0 0.5297061990576545 0.5297061990576545 0.5297061990576545

#overall number of boots (with 95% of transition point)
def cum_nbr_boots(li):
    '''from a list return a list of same size with the cumulative nbr of change in the values'''
    if len(li)==0:
        return []
    li = [True]+[li[i]!=li[i+1] for i in range(len(li)-1)]
    return(list(np.cumsum(li)))
#small exemple
#t = [1,1,1,1,2,2,2,3,3,3,4,4,2,2,2,2,3]
#cum_nbr_boots(t) #[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6]


#cumulative duration in zone x
def cum_duration_z(li, z):
    '''from a list return a list of same size with the cumulative duration in a certain zone'''    
    if len(li)==0:
        return []
    li = [k==z for k in li]
    return(list(np.cumsum(li)))
#small exemple
#t = [1,1,1,1,2,2,2,3,3,3,4,4,2,2,2,2,3]
#cum_duration_z(t,2) #[0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 7]
#cum_duration_z(t,1) #[1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


def density_mnlevel(li, nbr_value_per_bar, zone):
    '''from a list ourput a lsit of percentage of entry within a certain range with a specific zone'''
    #remove last entry to have a list divisible by nbr_value_per_bar
    li = li[:len(li)-len(li)%nbr_value_per_bar]
    #list of list of size nbr_value_per_bar, with true when its in the zone
    li = [sum([li[k]==zone for k in range(i, i+nbr_value_per_bar)])/nbr_value_per_bar for i in range(0,len(li),nbr_value_per_bar)]
    return(li)
#small exemple
#li = [1,1,1,1,2,2,2,3,3,3,4,4,2,2,2,2,1,1,1,3]
#density_mnlevel(li, 4, 1) #[1.0, 0.0, 0.0, 0.0, 0.75]
#density_mnlevel(li, 3, 2) #[0.0, 0.6666666666666666, 0.3333333333333333, 0.0, 1.0, 0.3333333333333333]

def vertical_travel_distance(li):
    v = [x[0] for x in itertools.groupby(li)]
    #replace '3_zone" by an integer
    v = [int(i.split('_')[0]) for i in v]
    #replace WG (zone 1) by zone 2 (cant be removed earlier as 3-1-3 would ber educed to 0 transtions for example)
    v = [int(str(i).replace('1','2')) for i in v]
    li_len = [abs(v[i+1]-v[i]) for i in range(0,len(v)-1)]
    return(sum(li_len))
#small example
#li = ['1_Zone','1_Zone','1_Zone','1_Zone',
#      '2_Zone','2_Zone','2_Zone',
#      '3_Zone',
#      '4_Zone','4_Zone','4_Zone','4_Zone',
#      '2_Zone','2_Zone','2_Zone','2_Zone',
#      '4_Zone','4_Zone','4_Zone','4_Zone']
#print(vertical_travel_distance(li)) #--> 6 (v=[1, 2, 3, 4, 2, 4], remove wg: v=[2, 3, 4, 2, 4], li_len=[1,1,2,2])
#li = ['1_Zone','1_Zone','1_Zone','1_Zone',
#      '2_Zone','2_Zone','2_Zone',
#      '3_Zone',
#      '4_Zone','4_Zone','4_Zone','4_Zone',
#      '1_Zone','1_Zone',
#      '4_Zone','4_Zone','4_Zone','4_Zone']
#vertical_travel_distance(li) #6    
    
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


def perc_element_list(li):
    c = Counter(li)
    return([(i, c[i] / len(li) * 100.0) for i in c])


def perc_element_dico(li):
    c = Counter(li)
    return {i: round(c[i] / len(li) * 100.0,3) for i in c}

#############

def ZoneVariable(df, config, timestamp_name='Timestamp', save=True, red_dot_for_each_hen=False, nbr_bird_per_square_meter=False, name_=''):

    '''From a dataframe of records, compute a Heatmap of number of birds in each zone at each 
    timestamp we are taking one value per minute (the first one), and we are not considering the rest
    red_dot_for_each_hen: if True, then we will plot where each bird is with a red dot in order to understand his synchronicity with other birds and if he likes crowd and when. It can then help extract some variables of interest
    nbr_bird_per_square_meter: If True, the nbr of birds will be divided by the umber of square meter associated to that zone'''
    
    #start recording the time it last
    START_TIME = time.perf_counter()
    
    #initialise variable
    path_extracted_data = config.path_extracted_data
    dico_zone_order = config.dico_zone_order
    id_run = config.id_run
    dico_zone_plot_name = config.dico_zone_plot_name
    
    print('----------------- Create time serie with each columns beeing a hen')
    df.sort_values(['Timestamp'], inplace=True)
    #use up to the second level only
    df['Timestamp'] = df['Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day,x.hour,x.minute, x.second))
    #remove the first record
    df = df.drop_duplicates(subset=['HenID','Timestamp'], keep='last')
    df_ts = time_series_henColumn_tsRow(df, config, col_ts='Zone', ts_with_all_hen_value=False, save=False, hen_time_series=False)

    #add date info
    df_ts['minute'] = df_ts[timestamp_name].map(lambda x: x.minute)
    df_ts['hour'] = df_ts[timestamp_name].map(lambda x: x.hour)
    df_ts['date'] = df_ts['Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day))
    
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
    path_ind = os.path.join(path_extracted_data,'visual','Nbr_bird_In_Zone','individuals')
    #creaindzte a director if not existing
    if not os.path.exists(path_ind):
        os.makedirs(path_ind)

    #for each day draw a heatmap
    for day in tqdm.tqdm(df_ts['date'].unique()):
        df_ = df_ts[df_ts['date']==day].sort_values([timestamp_name])
        #xaxis might be different over the days, if not complete days, so we will take the appropriate timestamp
        #take only the smallest timestamp per minute
        Xaxis = df_.groupby(['hour','minute'])[timestamp_name].agg(lambda x: min(list(x))).reset_index()[timestamp_name].tolist()       
        M = np.zeros(shape=(max(dico_zone_order.values())+1, len(Xaxis))) #+1 car starts from 0
        for i,ts in enumerate(Xaxis):
            #list of all zones happening on a particular timestamp that day
            li = list(df_[df_[timestamp_name]==ts][li_hen].values[0])
            c = Counter(li)
            #print(sum(list(c.values()))) 
            for zone_, order in dico_zone_order.items():
                if zone_ in c:
                    M[order][i] = c[zone_]
                    if nbr_bird_per_square_meter:
                        M[order][i] = M[order][i] / config.dico_zone_meter2[zone_]

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
            plt.savefig(os.path.join(path_,id_run+'_'+name_+'_'+plot_type+'_'+str(day).split('T')[0]+'.png'), format='png', dpi=300)
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
                li_zone_hen = df_[df_[timestamp_name].isin(Xaxis)][hen_].tolist()
                li_zone_hen = [dico_zone_order_[str(x)]+0.5 for x in li_zone_hen] #0.5 to show it in the middle of the heatmap bar
                ax.scatter(range(len(Xaxis)), li_zone_hen, marker='d', s=1, color='red') #s = size
                if save:
                    plt.savefig(os.path.join(path_ind,name_+'_'+hen_+'_'+plot_type+'_'+str(day).split('T')[0]+'.png'), 
                                format='png', dpi=300, bbox_inches='tight') 
                #plt.show()    
                plt.close()
                
    END_TIME = time.perf_counter()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))  
    
    
    
def DataRepresentation1(df_daily, config):
    START_TIME = time.perf_counter()
    pal_zone = config.pal_zone
    dico_zone_plot_name = config.dico_zone_plot_name
    path_extracted_data = config.path_extracted_data
    path_save = os.path.join(path_extracted_data, 'visual', 'data_representation','representation_color_and_duration')
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    # to save time computation we do all chicken together
    df_ = df_daily[['HenID','level','list_of_zones','list_of_durations']].copy()
    df_ = df_.dropna(how='any')
    df_ = df_[df_['list_of_durations']!='[]']
    for i in tqdm.tqdm(range(df_.shape[0])):
        x = df_.iloc[i]
        try:
            li_z = eval(x['list_of_zones'])
            li_dur = eval(x['list_of_durations'])
            HenID = x['HenID']
            level = x['level']
            entropy_of_duration = sampen(L=li_dur,r=0.2*np.std(li_dur), m=3)
            if len(li_dur)!=len(li_z):
                print('ERROR!!!')
                sys.exit()

            #define title 
            title = str(level).split(' ')[0]+' '+HenID
            #if already exists then dont do it
            #if len(glob.glob(os.path.join(path_save, id_run+'_'+title.replace(' ','_')+'.png')))==1:
            #    continue

            #plot
            y = [i/2 for i in li_dur] #cut in to as we want to plot half in negative and hal fin positive
            x = range(len(li_dur))
            fig = plt.figure()
            ax = plt.subplot(111)
            width=1
            ax.bar(range(len(x)), li_dur, width=width, color=[pal_zone[dico_zone_plot_name[z]] for z in li_z])
            ax.set_xticks(np.arange(len(x)) + width/2)
            ax.bar(range(len(x)), [-i for i in y], width=width, color=[pal_zone[dico_zone_plot_name[z]] for z in li_z])
            ax.set_xticks(np.arange(len(x)) + width/2)
            plt.axis('off') 
            plt.grid('off')
            plt.title(title+' entropy of duration:'+str(round(entropy_of_duration,2)))    
            plt.savefig(os.path.join(path_save, str(round(entropy_of_duration,2))+'_'+title.replace(' ','_')+'.png'), format='png')
            plt.close()
        except Exception as e:
            print('ERROR')
            print(e)
            return df_.iloc[i]
    END_TIME = time.perf_counter()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))  


def TimeSeriesPlot(df_ts, config, save=True, timestamp_name='New_Timestamp', last_folder_name='', name_=''):
    
    '''For a csv with one column=one time series, plot all the time series'''
    
    #start recording the time it last
    START_TIME = time.perf_counter()

    #initialise variable
    path_extracted_data = config.path_extracted_data
    dico_zone_order = config.dico_zone_order
    id_run = config.id_run

    li_hen = [i for i in df_ts.columns if i.startswith('hen_')]
    li_zones = list(df_ts[li_hen].stack().unique())
    df_ts['day'] = df_ts[timestamp_name].map(lambda x: dt.datetime(x.year,x.month,x.day)) 
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

    #for each hen draw a timeseries
    for hens in tqdm.tqdm(li_hen):
        fig, ax = plt.subplots(figsize=(25,3))
        df_plt = df_ts[~df_ts[hens].isnull()].sort_values([timestamp_name]).copy()
        zone_ts = df_plt[hens].map(lambda x: int(dico_zone_order[x])).tolist()
        plt.plot(df_plt[timestamp_name].tolist(), zone_ts)
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
        
    END_TIME = time.perf_counter()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))  

    
def TimeSeriesPlot_1row1day(df, config, save=True, timestamp_name='New_Timestamp', last_folder_name='', name_='', li_month=[],
                            col_ts='Zone'):
    
    '''For a csv with one column=one time series, plot all the time series, one row per day, one plot per month'''
    
    #start recording the time it last
    START_TIME = time.perf_counter()

    #into time serie
    df_ts = time_series_henColumn_tsRow(df, config, col_ts=col_ts, ts_with_all_hen_value=False, save=False, hen_time_series=False)
    display(df_ts.tail(3))
    print(df_ts.shape)
    #initialise variable
    path_extracted_data = config.path_extracted_data
    dico_zone_order = config.dico_zone_order
    id_run = config.id_run
    path_FocalBird = config.path_FocalBird
    date_max = config.date_max

    li_hen = [i for i in df_ts.columns if i.startswith('hen_')]
    li_zones = list(df_ts[li_hen].stack().unique())
    df_ts['day'] = df_ts[timestamp_name].map(lambda x: dt.datetime(x.year,x.month,x.day)) 
    df_ts['month'] = df_ts[timestamp_name].map(lambda x: x.month) 
    
    #remove the old month if asked
    if len(li_month)>0:
        df_ts = df_ts[df_ts['month'].isin(li_month)]

    #sort the yaxis
    #s = sorted(dico_zone_order.items(), key=operator.itemgetter(1))
    #li_zone_in_order = [x[0] for x in s]
    
    #create path where to save if not existing yet
    if last_folder_name=='':
        path_ = os.path.join(path_extracted_data,'visual','TimeSeriesPlot')
    else:
        path_ = os.path.join(path_extracted_data,'visual','TimeSeriesPlot',last_folder_name)
    #create a director if not existing
    if not os.path.exists(path_):
        os.makedirs(path_) 

    #add basics hens info
    #download info on henID associtation to (TagID,date) 
    df_FB = pd.read_csv(path_FocalBird, sep=',', parse_dates=['StartDate','EndDate'], dayfirst=True, encoding='latin') 
    df_FB['HenID'] = df_FB['HenID'].map(lambda x: 'hen_'+str(x))
    df_FB = df_FB[df_FB['ShouldBeExcluded']!='yes']
    df_FB['EndDate'].fillna(date_max+dt.timedelta(days=1), inplace=True)
    #create a dictionary with henID as keys and a list of tracking-active days
    dico_hen_activedate = defaultdict(list)
    for i in range(df_FB.shape[0]):
        x = df_FB.iloc[i]
        li_dates = pd.date_range(start=x['StartDate']+dt.timedelta(days=1), 
                                 end=x['EndDate']-dt.timedelta(days=1), freq='D')
        dico_hen_activedate[x['HenID']].extend([dt.datetime.date(d) for d in li_dates])
        
    #for each hen draw a timeseries per day, one plot per month
    for hens in tqdm.tqdm(list(reversed(li_hen))):
        df_plt = df_ts[~df_ts[hens].isnull()].sort_values([timestamp_name]).copy()
        df_plt[hens] = df_plt[hens].map(lambda x: int(dico_zone_order[x]))
        
        #### remove days that should not be taken into account for this hen
        df_plt['day'] = df_plt['day'].map(lambda x: dt.datetime(x.year, x.month, x.day)) 
        df_plt['date_2remove_penhen'] = df_plt.apply(lambda x: x['day'] not in dico_hen_activedate[hens], axis=1)
        print('TEST ', df_plt[df_plt['date_2remove_penhen']].shape)
        df_plt = df_plt[~df_plt['date_2remove_penhen']]        
        
        for month,df_plt_ in df_plt.groupby('month'):
            li_day = df_plt_['day'].unique()
            l = len(li_day) ; c = 1
            fig = plt.figure(figsize=(c*5, l*1))
            for i,d in enumerate(li_day):
                plt.subplot(l,c,i+1)
                df_plt__ = df_plt_[df_plt_['day']==d].copy()
                plt.plot(df_plt__[timestamp_name].tolist(), df_plt__[hens].tolist())
                plt.xticks(fontsize=4)
                plt.yticks(fontsize=4)
                plt.title(str(d).split('T')[0], size=4)
            #plt.title(str(month)+'_'+hens)
            if save:
                plt.savefig(os.path.join(path_,id_run+'_ts_'+name_+'_month'+str(month)+'_'+hens+'.png'), format='png',
                            bbox_inches='tight', dpi=300) #dpi=300 is necessary to read the dates and hours, but two times longer!!
            #plt.show();
            plt.clf()
            plt.close("all")
            gc.collect()
        
    END_TIME = time.perf_counter()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))  
    
def is_day(x, dico_):
    '''from a timestamp value x, and the dico_night_hour parameter, it will output true if its during the day, false otherwise'''
    if max(dico_.keys())<dt.datetime(x.year,x.month,x.day,0,0,0):
        print('ERROR: your \"dico_nighthour\" parameter does not include information for the date: %s'%str(x))
        sys.exit()
    else:
        #take info (i.e. values) of the dico_night_hour key that represent the smallest date among all the date>=x:
        m = min([d for d in dico_.keys() if d>=dt.datetime(x.year,x.month,x.day,0,0,0)])
        #is the timestamp smaller than the latest day hour and bigger than the first day hour? 
        #Attention-limitation: midnight should not be included in the day hour
        return((dt.datetime(1,1,1,x.hour,x.minute,0)>=dt.datetime(1,1,1,dico_[m]['start_h'],dico_[m]['start_m'],0)) & \
        (dt.datetime(1,1,1,x.hour,x.minute,0)<dt.datetime(1,1,1,dico_[m]['end_h'],dico_[m]['end_m'],0)))

def nbr_device_info(x):
    '''due to how our tracking system was made, it changed over time!'''
    if x<=dt.datetime(2020,10,29,0,0,0):
        return [72-1]
    elif x<=dt.datetime(2020,11,5,0,0,0):
        return [78-1]
    elif x<=dt.datetime(2020,11,12,0,0,0):
        return [84-1]
    elif x<=dt.datetime(2021,7,26,0,0,0):
        return [90-1,84]
    else:
        return [np.nan]    
    

def nbrh_WG_open(x, config):
    dico_garden_opening_hour = config.dico_garden_opening_hour
    close_dates = config.close_dates
    date_first_opening_WG = config.date_first_opening_WG
    if (x<date_first_opening_WG) | (x in close_dates):
        return 0
    #take info (i.e. values) of the dico_ key that represent the smallest date among all the date>=x:
    m = min([d for d in dico_garden_opening_hour.keys() if d>=dt.datetime(x.year,x.month,x.day,0,0,0)])
    v = dico_garden_opening_hour[m]
    return ((v['end_h'] - v['start_h'])*60 - v['start_m'] + v['end_m'])/60   
#small example
#nbrh_WG_open(config,dt.datetime(2020,10,10)) #4h30 
    
def is_WG_open(x, config, epsi_open=0, epsi_close=20):
    '''from a timestamp value x, the dico_ (typically: dico_garden_opening_hour) and the date_first_opening_WG parameters, 
    it will output true if the WG is open, false otherwise.
    With the epsi_* parameters it allows to be more flexible with the true time of opening/closing'''
    
    dico_ = config.dico_garden_opening_hour
    date_first_opening_WG = config.date_first_opening_WG
    close_dates = config.close_dates
    
    #if no record return nan
    if pd.isnull(x)==True:
        return(np.nan)
    if (x<date_first_opening_WG) | (dt.datetime(x.year,x.month,x.day) in close_dates):
        return(False)
    if max(dico_.keys())<dt.datetime(x.year,x.month,x.day,0,0,0):
        print('ERROR: your \"dico_garden_opening_hour\" parameter does not include information for the date: %s'%str(x))
        sys.exit()
    else:
        #take info (i.e. values) of the dico_ key that represent the smallest date among all the date>=x:
        m = min([d for d in dico_.keys() if d>=dt.datetime(x.year,x.month,x.day,0,0,0)])
        #is the timestamp bigger than the first day hour-epsi_open & smaller than the latest day hour+epsi_close:
        return (dt.datetime(1,1,1,x.hour,x.minute,0)>=(dt.datetime(1,1,1,dico_[m]['start_h'], dico_[m]['start_m'],0)-\
                                                       dt.timedelta(minutes=epsi_open))) & \
               (dt.datetime(1,1,1,x.hour,x.minute,0)<(dt.datetime(1,1,1, dico_[m]['end_h'], dico_[m]['end_m'],0)+\
                                                       dt.timedelta(minutes=epsi_close)))
    
    
def is_ts_before_MN_opening(x, dico_, MN, date_first_opening_WG, close_dates, epsi_open=0):
    '''from a timestamp value x, and the dico_garden_opening_hour parameter, it will output true if the WG is open, 
    false otherwise.
    With the epsi_* paramteres it allows to be more flexible with the true time of opening.
    Note: the difference with is_WG_open() fct, is that the end_h, end_mn is here start_h,start_mn+mn, and that before the WG 
    started to be opened, we return np.nan, and not false.'''
    #if no record return nan
    if pd.isnull(x)==True:
        return(np.nan)
    #return nan if the WG was not open, as "False" would be wrong
    if (x<date_first_opening_WG) | (x in close_dates):
        return(np.nan)
    if max(dico_.keys())<dt.datetime(x.year,x.month,x.day,0,0,0):
        print('ERROR: your \"dico_garden_opening_hour\" parameter does not include information for the date: %s'%str(x))
        sys.exit()
    else:
        #take info (i.e. values) of the dico_ key that represent the smallest date among all the date>=x:
        m = min([d for d in dico_.keys() if d>=dt.datetime(x.year,x.month,x.day,0,0,0)])
        #is the timestamp bigger than the first day hour-epsi_open & smaller than first day hour+MN:
        return (dt.datetime(1,1,1,x.hour,x.minute,0)>=(dt.datetime(1,1,1,dico_[m]['start_h'], dico_[m]['start_m'],0)-\
                                                       dt.timedelta(minutes=epsi_open))) & \
               (dt.datetime(1,1,1,x.hour,x.minute,0)<(dt.datetime(1,1,1, dico_[m]['start_h'],dico_[m]['start_m'],0)+\
                                                      dt.timedelta(minutes=MN)))

    
#small examples
#print(is_WG_open(dt.datetime(2020,11,1,14,0,5),dico_, date_first_opening_WG, epsi_open=1, epsi_close=20), 
#is_WG_open(dt.datetime(2020,10,7,14,0,5),dico_, date_first_opening_WG, epsi_open=1, epsi_close=20),
#is_WG_open(dt.datetime(2020,10,8,10,59,59),dico_, date_first_opening_WG, epsi_open=1, epsi_close=20),
#is_WG_open(dt.datetime(2020,10,8,10,59,59),dico_, date_first_opening_WG, epsi_open=0, epsi_close=20),
#is_WG_open(dt.datetime(2020,10,8,11,0,0),dico_, date_first_opening_WG, epsi_open=1, epsi_close=20))
#True False True False True
#print(is_ts_before_MN_opening(dt.datetime(2020,11,1,14,0,5),dico_, 15, date_first_opening_WG, epsi_open=1), 
#is_ts_before_MN_opening(dt.datetime(2020,10,7,14,0,5),dico_, 15, date_first_opening_WG, epsi_open=1),
#is_ts_before_MN_opening(dt.datetime(2020,10,8,10,59,59),dico_, 15, date_first_opening_WG, epsi_open=1),
#is_ts_before_MN_opening(dt.datetime(2020,10,8,11,14,0),dico_, 15, date_first_opening_WG, epsi_open=1),
#is_ts_before_MN_opening(dt.datetime(2020,10,8,11,15,0),dico_, 15, date_first_opening_WG, epsi_open=1))
#False nan True True False

    
def correct_key(x,dico_night_hour):
    '''from a specific timestamp and a dico_night_hour as inputs, it will output the key of the dico_night_hour that is associated to the
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
                        dico_night_hour[correct_key(t1,dico_night_hour)]['end_h'],
                        dico_night_hour[correct_key(t1,dico_night_hour)]['end_m'],0):
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
    return(dt.datetime(1,1,1,x.hour,x.minute,0)>=dt.datetime(1,1,1,dico_night_hour[m]['end_h'],
                                                     dico_night_hour[m]['end_m'],0))    


def min_date_nestbox(x, nestbox_sec):
    if len(str(x))>3:
        return(min([k for k,l in x if float(l)>=nestbox_sec], default=np.nan))
    return np.nan
#small example
#print(min_date_nestbox([(dt.datetime(2020,11,1,6,0,0),15*60),(dt.datetime(2020,11,1,5,30,0),14*60+1)],15*60), #2020-11-01 06:00:00 
#min_date_nestbox([(dt.datetime(2020,11,1,6,0,0),15*60-1),(dt.datetime(2020,11,1,5,30,0),14*60+1)],15*60)) #nan    

def nbr_visit_and_longeststay(h,x,type_):
    '''e.g. x: "[(Timestamp('2020-09-30 09:41:13'), '5.0'), (Timestamp('2020-09-30 11:28:07'), '3.0'),
    (Timestamp('2020-09-30 16:51:25'), '517.0')]"'''
    if type_ not in ['before','after']:
        print('ERROR: parameter type_ must be either before or after')
        sys.exit()
    if len(str(x))<=3:
        return (0,0)
    if type_=='before':
        #retrieve all visits before or equal to that time
        li = [(t,d) for t,d in x if t<=dt.datetime(t.year,t.month,t.day,h,0,0)]
    if type_=='after':
        #retrieve all visites after or equal to that time
        li = [(t,d) for t,d in x if t>=dt.datetime(t.year,t.month,t.day,h,0,0)]
    #return a tuple with number of visits that started before that time, and the longest duration
    return (len(li), max([float(d) for t,d in li], default=0))
    
    
def successfullIntrusion(h,x,nestbox_sec):
    #list of intrustion before that hour on the same day
    if len(str(x))<=3:
        return np.nan
    li = [(t,d) for t,d in x if t<=dt.datetime(t.year,t.month,t.day,h,0,0)]
    #return the successful intrusion ratio: (#staid <h longer than nestbox_sec mn) / (#of staid <h)
    if len(li)>0:
        return(len([t for t,d in li if float(d)>=nestbox_sec])/len(li))
    return 0

def WG_open_time(config, x):
    dico_ = config.dico_garden_opening_hour
    #take info (i.e. values) of the dico_ key that represent the smallest date among all the date>=x:
    m = min([d for d in dico_.keys() if d>=dt.datetime(x.year,x.month,x.day,0,0,0)])
    t = dico_[m]
    return (dt.datetime(1999,1,1,t['end_h'],t['end_m'],0) - dt.datetime(1999,1,1,t['start_h'],t['start_m'],0)).total_seconds()

def WG_open_dayduration_untilopen(config, x):
    dico_ = config.dico_garden_opening_hour
    #take info (i.e. values) of the dico_ key that represent the smallest date among all the date>=x:
    m = min([d for d in dico_.keys() if d>=dt.datetime(x.year,x.month,x.day,0,0,0)])
    t = dico_[m]
    return (dt.datetime(1999,1,1,t['start_h'],t['start_m'],0)-dt.datetime(1999,1,1,0,0,0)).total_seconds()


#https://en.wikipedia.org/wiki/Sample_entropy
#" SampEn is the negative natural logarithm of the probability that if two sets of simultaneous data points of length 
#{\displaystyle m}m have distance {\displaystyle <r}<r then two sets of simultaneous data points of length {\displaystyle m+1}
#m+1 also have distance {\displaystyle <r}<r."
#A smaller value of {\displaystyle SampEn}SampEn also indicates more self-similarity in data set or less noise.
def sampen(L, r=2, m=2):
    '''Usually m=2 and r=0.2*std'''
    N = len(L)
    B = 0.0
    A = 0.0
    
    # Split time series and save all templates of length m
    xmi = np.array([L[i:i+m] for i in range(N-m)])
    xmj = np.array([L[i:i+m] for i in range(N-m+1)])
       
    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii-xmj).max(axis=1) <= r)-1 for xmii in xmi])
            
    # Similar for computing A
    m += 1
    xm = np.array([L[i:i+m] for i in range(N-m+1)])
        
    A = np.sum([np.sum(np.abs(xmi-xm).max(axis=1) <= r)-1 for xmi in xm])
    
    # Return SampEn
    return -np.log(A/B)




def HenDailyVariable_Origins(df, config, name_='', timestamp_name='Timestamp', save=True, has_cons_equal_zone=True): 
    
    ''' 
    Note: work with ts that have nan (typically at begining)
    
    Input:
    df_ts: Each row correspond to a specific timestamp, each column to a specific hen timeseries (which column name must start 
        with hen_ ). Must also have a Timestamp and a level column, which will be used to aggregate info and compute variables on these 
        aggregated info
    config: file with parameter
    has_cons_equal_zone: if the initial data has some consecutives euqal zone for hte same hen (that are not necessarily at the same time)
    
    Output:
    Dataframe with according variables'''
    
    #start recording the time it last
    START_TIME = time.perf_counter()
    
    #remove milliseconds now that we cleaned the data (i.e. the records with less than 1seconds duration
    #sort by timestamp
    df.sort_values([timestamp_name], inplace=True)
    #use up to the second level only
    df[timestamp_name] = df[timestamp_name].map(lambda x: dt.datetime(x.year,x.month,x.day,x.hour,x.minute, x.second))
    #remove the first record
    df = df.drop_duplicates(subset=['HenID',timestamp_name], keep='last')
    
    #remove duration if existing in the dataframe to avoid error
    if 'duration' in df.columns:
        df.drop('duration', axis=1, inplace=True)
    df_init = df.copy()
    print('----------------- Create time serie')
    df_ts = time_series_henColumn_tsRow(df, config, col_ts='Zone', ts_with_all_hen_value=False, save=False, hen_time_series=False)
    
    #compute nbr_sec computation here (list of difference between each timestamp, and must always be the same)
    li_ts = df_ts[timestamp_name].tolist()
    li_diff_ts = list(set(list(map(operator.sub, li_ts[1:], li_ts[0:-1]))))
    if len(li_diff_ts)!=1:
        print('ERROR: your timestamp columns have different one to one difference: ', li_diff_ts)
        sys.exit()
    nbr_sec = li_diff_ts[0].total_seconds()
    print('your time series has %d seconds between two timestamps'%nbr_sec)    
    
    ############ initialise parameters from config file
    path_extracted_data = config.path_extracted_data
    id_run = config.id_run
    date_max = config.date_max
    dico_night_hour = config.dico_night_hour
    dico_zone_order = config.dico_zone_order
    date_first_opening_WG = config.date_first_opening_WG
    close_dates = config.close_dates
    dico_garden_opening_hour = config.dico_garden_opening_hour
    path_FocalBird = config.path_FocalBird
    path_Days = config.path_Days
    nestbox_sec = config.nestbox_sec
    ANestboxHour = config.ANestboxHour
    BNestboxHour = config.BNestboxHour
    successfullIntrusionHour = config.successfullIntrusionHour
    WG_after_opening_mn = config.WG_after_opening_mn
    NonTrans_dur_sec = config.NonTrans_dur_sec
    nbr_sec_chaoticmvt_notmiddle = config.nbr_sec_chaoticmvt_notmiddle
    li_nbr_sec_chaoticmvt_middle = config.li_nbr_sec_chaoticmvt_middle
    li_perc_activity = config.li_perc_activity
    #EntropyTimeComputation = config.EntropyTimeComputation
    #NbrData = config.NbrData
    lf_counter = config.lf_counter
    
    ############ small verifications
    #verify columns name of df_ts and select the column we need
    li_hen = [i for i in list(df_ts) if i.startswith('hen_')]
    #verify that the timestamp has same difference than the suggested nbr_sec parameter
    df_ts = df_ts.sort_values(timestamp_name)
    if (df_ts[timestamp_name].iloc[1]-df_ts[timestamp_name].iloc[0]).seconds!=nbr_sec:
        print('ERROR: your timestamp difference does not equal your nbr_sec parameter')
        sys.exit()
        
    ############ separate day and night
    df_ts['is_day'] = df_ts[timestamp_name].map(lambda x: is_day(x, dico_night_hour))
    #night
    df_ts_night = df_ts[~df_ts['is_day']].copy()
    df_ts_night['night_level'] = df_ts_night[timestamp_name].map(lambda x: str(x)[0:-9]+'_'+str(x+dt.timedelta(days=1))[8:10] if\
                                            name_level(x,dico_night_hour) else str(x-dt.timedelta(days=1))[0:-9]+'_'+str(x)[8:10])
    #days
    #note that minuit is: 0, and its date should be as 1,2 (day-1, day)
    df_ts = df_ts[df_ts['is_day']].copy()
    df_ts['level'] = df_ts['date'].copy()
    
    ############ open days to be removed
    df_day = pd.read_csv(path_Days, sep=',', parse_dates=['Date'], dayfirst=True) #date verified: correct

    
    ########################################################
    #### night info 
    ########################################################    
    #have to be done at begining to free memory space
    print('----------------- main night zone and nbr of transitions over night....')
    df_ts_night = pd.melt(df_ts_night.filter([timestamp_name,'night_level']+li_hen), id_vars=[timestamp_name,'night_level'],
                          value_vars=li_hen)
    df_ts_night.rename(columns={'variable':'HenID','value':'Zone'}, inplace=True)
    df_ts_night = df_ts_night[~df_ts_night['Zone'].isnull()].groupby(['HenID','night_level']).agg(
                            night_Max_duration_zones=pd.NamedAgg(column='Zone', aggfunc=lambda x: max_duration_zones(x)),
                            night_duration_Z5=pd.NamedAgg(column='Zone', aggfunc=lambda x: duration_Z5(x)),
                            night_Total_number_transition=pd.NamedAgg(column='Zone', 
                                                                      aggfunc=lambda x: nbr_transition(list((x))))).reset_index()
    df_ts_night['is_mvt_night'] = df_ts_night['night_Total_number_transition'].map(lambda x: int(x>0))
    
    #amount of transition per hour during the night
    df_n = df_init.copy()
    df_n['is_day'] = df_n['Timestamp'].map(lambda x: is_day(x, dico_night_hour))
    df_n['night_level'] = df_n['Timestamp'].map(lambda x: str(x)[0:-9]+'_'+str(x+dt.timedelta(days=1))[8:10] if\
                                            name_level(x,dico_night_hour) else str(x-dt.timedelta(days=1))[0:-9]+'_'+str(x)[8:10])
    df_n['hour'] = df_n['Timestamp'].map(lambda x: x.hour)
    
    #useless we dont want it
    #if not has_cons_equal_zone:
    #    #ATTENTION: consecutives records of same zone should be removed, otherwise it will be counted twice here!!
    #    df_n_ = df_n[~df_n['is_day']].groupby(['HenID','night_level','hour'])['Timestamp'].count().reset_index()
    #    df_n__ = df_n_.pivot(index=['HenID','night_level'], columns='hour', values='Timestamp').reset_index()
    #    df_n__.rename(columns={i:'nbr_transition_at_h'+str(i) for i in df_n_['hour'].unique()}, inplace=True)
    #    df_n__= df_n__.fillna(0)
    #    #first, create the nbr_transition_next1hafterlightoff
    #    df_n__['start_night_hour'] = df_n__['night_level'].map(lambda x: dico_night_hour[min([d for d in dico_night_hour.keys() if\
    #                                                                       d>=dt.datetime.strptime(x.split('_')[0],'%Y-%m-%d')])]['end_h'])
    #    #display(df_n__[['night_level','start_night_hour']])
    #    df_n__['nbr_transition_next1hafterlightoff'] = df_n__.apply(lambda x: x['nbr_transition_at_h'+str(x['start_night_hour'])], axis=1)
    #    #display(df_n__['nbr_transition_next1hafterlightoff'].value_counts())
    #    #then, remove the 17h, 18h, 19h, 20h hour due to change of winter/summer time
    #    #we drop 'start_night_hour, as any way this will only appear in the ne that had a mvt, and its not of interest
    #    df_n__.drop(['nbr_transition_at_h17', 'nbr_transition_at_h18', 'nbr_transition_at_h19','nbr_transition_at_h20','start_night_hour'],
    #                inplace=True, axis=1) 
    #    df_ts_night = pd.merge(df_ts_night, df_n__, on=['HenID','night_level'], how='left')
        
    df_ts_night = df_ts_night.fillna(0) #add 0 to all the one that had no transition over night
    df_ts_night['level'] = df_ts_night['night_level'].map(lambda x: dt.datetime.strptime(x.split('_')[0], '%Y-%m-%d'))
    
    ########################################################    
    ############ one row per unique hen-timestamp 
    ########################################################    
    df = df_ts.filter([timestamp_name,'level']+li_hen).copy()  
    #list of involved level
    li_day = set(df['level'].tolist())  
    df = pd.melt(df, id_vars=[timestamp_name,'level'], value_vars=li_hen)
    df.rename(columns={'variable':'HenID','value':'Zone'}, inplace=True)
    #we define the duration of each row to be the nbr_sec, its better than computing with the next timestamp as if we removed some days
    #due to health-assessemnt, then it will induce wrong durations! also more efficient that way. BUT its an assumption, that the row must
    #be equally spaced and nbr_sec is the duration in between each timestamp
    df['duration_sec'] = nbr_sec
    #list of not nan Zones
    li_Zone = [x for x in df[~df['Zone'].isnull()]['Zone'].unique()]

    
    ########################################################
    print('----------------- total & percentage duration per Zone in seconds....')
    #one row per day, hen, existingzone
    df_ = df.groupby(['HenID','level','Zone'])['duration_sec'].agg(lambda x: sum(x)).reset_index()
    #one row per day and hen, each columns account for a zone_duration
    df_daily = df_.pivot_table(values='duration_sec', index=['HenID', 'level'], columns='Zone')
    df_daily.rename(columns={x:'duration_'+x for x in li_Zone}, inplace=True)
    #lets verify with total duration
    df_daily['verification_daily_total_duration'] = df_daily.apply(lambda x: np.nansum([x[i] for i in ['duration_'+x for x in li_Zone]]),
                                                                   axis=1)
    df_daily['verification_daily_total_nbr_hour'] = df_daily['verification_daily_total_duration'].map(lambda x: x/60/60)

    df_daily = df_daily.reset_index()
    #replace np.nan duration by 0
    df_daily.replace(np.nan,0, inplace=True)
    #print('The number of hours per \"level\" period is of:')
    #display(df_daily.groupby(['verification_daily_total_nbr_hour'])['level','HenID'].agg(lambda x: list(x)).reset_index())

    #create an ordered list of the normalized duration per zone for chi2distance later (hen will first be sorted by entropy, and 
    #hence we will do this at the end)
    li_zone_dur = [c for c in df_daily.columns if c.startswith('duration_')] #keep same order
    df_daily['dur_values'] = df_daily.apply(lambda x: str([x[i] for i in li_zone_dur]), axis=1)
    df_daily['dur_values'] = df_daily['dur_values'].map(lambda x: eval(x))
    df_daily['dur_values_normalized'] = df_daily['dur_values'].map(lambda x: [i/float(np.sum(x)) if float(np.sum(x))!=0 else 0 for i in x])

    #add percentage in each zone also in zone 1
    for c in ['duration_'+x for x in li_Zone]:
        df_daily['perc_'+c] = df_daily.apply(lambda x: round(x[c]/x['verification_daily_total_duration']*100), axis=1)
    #(Total time outside)/(total time wg is open)
    df_daily['time_wg_open_sec'] = df_daily['level'].map(lambda x: WG_open_time(config,x)) #duratio of WG opening
    df_daily['duration_sincedaystarted_beforeWGopened_sec'] = df_daily['level'].map(lambda x: WG_open_dayduration_untilopen(config,x))
    df_daily['perc_1_Zone_while_WG_open'] = df_daily.apply(lambda x: x['duration_1_Zone']/x['time_wg_open_sec']*100, axis=1)
    #replace by np.nan when WG is close
    df_daily.loc[df_daily['level']<date_first_opening_WG,'perc_1_Zone_while_WG_open'] = np.nan

    ########################################################
    print('----------------- first timestamp in each zone per day & latency var....')
    #why: will be usefull to produce other variables, to verify the code and to use it for some zones
    df_ = df.groupby(['HenID', 'level','Zone'])[timestamp_name].agg(lambda x: min(list(x))).reset_index()
    #agg function = 'first' ats its string value, and the default function is the mean. Here by construction df_ has unique such 
    #values
    df__ = df_.pivot_table(values=timestamp_name, index=['HenID', 'level'], columns='Zone', aggfunc='first')
    df__.rename(columns={x:'FirstTimestamp_'+x for x in li_Zone}, inplace=True)
    df__ = df__.reset_index()
    df_daily = pd.merge(df_daily, df__, how='outer', on=['HenID','level'])
    #add latency variables
    for x in li_Zone:
        v = 'FirstTimestamp_'+x
        df_daily[v.replace('FirstTimestamp','latency')+'_h'] = df_daily[v].map(lambda x: (dt.datetime(x.year,x.month,x.day,
                                                                      dico_night_hour[correct_key(x, dico_night_hour)]['end_h'],
                 dico_night_hour[correct_key(x, dico_night_hour)]['end_m'],0)-x).total_seconds()/60/60 if x is not pd.NaT else 0)
        
    #latency in WG
    df_daily['FirstTimestamp_1_Zone_sec'] = df_daily['FirstTimestamp_1_Zone'].map(lambda x: (x.hour*60*60+x.minute*60+x.second))
    df_daily['latency_1_Zone_h'] = df_daily.apply(lambda x: (x['FirstTimestamp_1_Zone_sec']-
                                                            x['duration_sincedaystarted_beforeWGopened_sec'])/60/60 if x is not pd.NaT else 0, axis=1)    
    #replace by np.nan when WG is close
    df_daily.loc[df_daily['level']<date_first_opening_WG,'latency_1_Zone_h'] = np.nan


    ########################################################
    print('----------------- number of Zone (excluding nan)....')
    df_ = df[~df['Zone'].isnull()].groupby(['HenID','level'])['Zone'].agg(lambda x: len(set((x)))).reset_index()
    df_.rename(columns={'Zone':'Total_number_zone'}, inplace=True)
    df_daily = pd.merge(df_daily, df_, how='outer', on=['HenID','level'])
    
        
    ########################################################        
    #compute some variables 
    ########################################################
    #based on a list of zones over a day, where each zone count for the same nbr_sec second
    #e.g.[einstreu,eintreu,rampe,rampe.....]
    #excluding empty zones, because it influences for exemple the entropy computation (if full of nan, then might be more predictable)    
    print('----------------- compute some variables based on a list of zones over a day....')
    #note that entropy gets as input a list of proba distribution (or count of each element)                    
    df_ = df[~df['Zone'].isnull()].groupby(['HenID','level']).agg(
           list_of_durations=pd.NamedAgg(column='Zone', aggfunc=lambda x: list_of_durations(x, nbr_sec)),
           food_related_behavior=pd.NamedAgg(column='Zone', aggfunc=lambda x: food_related_behavior(li_Z=list(x), config=config)),
           food_related_behavior_rp=pd.NamedAgg(column='Zone', aggfunc=lambda x: food_related_behavior(li_Z=list(x), config=config, 
                                                                                                       fake_sec=20*60)),
           food_related_behavior_rm=pd.NamedAgg(column='Zone', aggfunc=lambda x: food_related_behavior(li_Z=list(x), config=config,
                                                                                                      fake_sec=-20*60)),
           nestboxes_related_behavior=pd.NamedAgg(column='Zone', aggfunc=lambda x: nestboxes_related_behavior(li=list(x), config=config)),
           zone_list=pd.NamedAgg(column='Zone', aggfunc=lambda x: tuple(x)),
           list_of_zones=pd.NamedAgg(column='Zone', aggfunc=lambda x: list_of_zones(list(x))),
           Max_duration_zones=pd.NamedAgg(column='Zone', aggfunc=lambda x: max_duration_zones(x)),
           dico_duration_stats=pd.NamedAgg(column='Zone', aggfunc=lambda x: dico_duration_stats(x, nbr_sec)),
           dico_zone_sortedduration=pd.NamedAgg(column='Zone', aggfunc=lambda x: dico_zone_sortedduration(x, nbr_sec)),
           Total_number_transition=pd.NamedAgg(column='Zone', aggfunc=lambda x: nbr_transition(list(x))),
           nbr_stays=pd.NamedAgg(column='Zone', aggfunc=lambda x: nbr_bouts_per_zone(list(x))),
           nbr_appearances=pd.NamedAgg(column='Zone', aggfunc=lambda x: \
                                        dict_of_zones_appearances_with_transitionalZones([int(i.split('_Zone')[0]) for i in list(x)])), 
           empproba=pd.NamedAgg(column='Zone', aggfunc=lambda x: \
                                empirical_probabilites_of_goingup([int(i.split('_Zone')[0]) for i in list(x)])),
           distribution_entropy=pd.NamedAgg(column='Zone', aggfunc=lambda x: entropy(list(Counter(list(x)).values()),base=2)),
           vertical_travel_distance=pd.NamedAgg(column='Zone', aggfunc=lambda x: vertical_travel_distance(list(x))),
           #SampEnt_order2=pd.NamedAgg(column='Zone', aggfunc=lambda x: sample_entropy([dico_zone_order[i] for i in x], order=2,
           #                                                                                metric='chebyshev')),
           t_DU_missingZone_mvtPerc=pd.NamedAgg(column='Zone', aggfunc=lambda x: li_missingZone_mvtPerc_DU([dico_zone_order[i] for i in x],
                                                                                                                nbr_sec)),
           li_event_chaoticmvt_z_d=pd.NamedAgg(column='Zone', aggfunc=lambda x: li_event_chaoticmvt_z_d([dico_zone_order[i] for i in x],
                                                                              nbr_sec, nbr_sec_chaoticmvt_notmiddle, dico_zone_order))
           ).reset_index()

    df_daily = pd.merge(df_daily, df_, how='outer', on=['HenID','level'])
    
    #add sample entropy without accounting on duration
    df_daily['SampleEntropy'] = df_daily['list_of_zones'].map(lambda x: sampen([config.dico_zone_order[i] for i in x],
                                                                               r=0,m=2) if len(x)>5 else np.nan)

    #total number of stay
    df_daily['nbr_stays_total'] = df_daily['nbr_stays'].map(lambda x: np.nansum(list(x.values())))
    #retrieve info per zone
    for z in li_Zone:
        df_daily['nbr_stays_'+z] = df_daily['nbr_stays'].map(lambda x: x.get(z,0))
        df_daily['perc_nbr_stays_'+z] = df_daily.apply(lambda x: x['nbr_stays_'+z]/x['nbr_stays_total']*100, axis=1)
        df_daily['nbr_appearances_'+z] = df_daily['nbr_appearances'].map(lambda x: x.get(int(z.split('_')[0]),0))
        df_daily['empproba_'+z] = df_daily['empproba'].map(lambda x: x.get(z,0)) #if no transition to a zone then emp proba is 0
    #total number of stay %of stays per zone
    df_daily['nbr_stays_total'] = df_daily['nbr_stays'].map(lambda x: np.nansum(list(x.values())))

    df_daily.drop(['nbr_stays','empproba','nbr_appearances'], inplace=True, axis=1)
    #add maximum duration in zone4
    df_daily['Max_duration_zone_4'] = df_daily['dico_zone_sortedduration'].map(lambda x: max(x.get('4_Zone',[0])))
    
    #add missingzone percentage
    df_daily['down_missingZone_mvtPerc'] = df_daily['t_DU_missingZone_mvtPerc'].map(lambda x: x[0])
    df_daily['up_missingZone_mvtPerc'] = df_daily['t_DU_missingZone_mvtPerc'].map(lambda x: x[1])
    #df_daily['down_missingZone_mvtNbr'] = df_daily['t_DU_missingZone_mvtPerc'].map(lambda x: x[0]*(x[2]-1)/100)
    #df_daily['up_missingZone_mvtNbr'] = df_daily['t_DU_missingZone_mvtPerc'].map(lambda x: x[1]*(x[2]-1)/100)
    #df_daily[['down_missingZone_mvtPerc','up_missingZone_mvtPerc']].fillna(0,inplace=True)
    #add info from stats of the duration list (from a dictionary column into x (=len(dico)) columns)
    df_daily = pd.concat([df_daily.drop(['dico_duration_stats'], axis=1), df_daily['dico_duration_stats'].apply(pd.Series)], axis=1)
    df_daily.loc[df_daily['Total_number_transition']<=10, 'kurtosis_duration'] = np.nan
    df_daily.loc[df_daily['Total_number_transition']<=10, 'skew_duration'] = np.nan
    
    #Number of stayed longer than NonTransitional_dur seconds (to remove transitional zones) in each zone 
    #total
    df_daily['stay_longer_'+str(NonTrans_dur_sec)+'sec'] = df_daily['list_of_durations'].map(lambda x: len([i for i in x if i>=NonTrans_dur_sec]))
    #per zone
    for z in li_Zone:
        df_daily['stay_longer_'+str(NonTrans_dur_sec)+'sec_'+z] = df_daily['dico_zone_sortedduration'].map(lambda x: len([i for i in x.get(z,[0])if i>=NonTrans_dur_sec]))

    #add info on chaoticmvt
    df_daily['dico_z_chaoticmvtMiddleDuration'] = df_daily['li_event_chaoticmvt_z_d'].map(lambda x: stats_chaoticmvt(x))
    #to avoid warning induced by "A Series of dtype=category constructed from an empty dict will now have categories of 
    #dtype=object rather than dtype=float64, consistently with the case in which an empty list is passed "
    df_daily['dico_z_chaoticmvtMiddleDuration'] = df_daily['dico_z_chaoticmvtMiddleDuration'].map(lambda x: np.nan if x=={} else x)
    df_daily = pd.concat([df_daily, df_daily['dico_z_chaoticmvtMiddleDuration'].apply(pd.Series)], axis=1)
    #when np.nan, then change to be 0, a
    for c in [x for x in df_daily.columns if str(x).startswith('chaoticmvt_Middle')]:
        df_daily[c] = df_daily[c].fillna(0)
        for nbr_ in li_nbr_sec_chaoticmvt_middle:
            df_daily[c+'_nbr_'+str(nbr_)+'mn'] = df_daily[c].map(lambda x: len([i for i in x if i<=nbr_]) if x!=0 else 0)
            
    ########################################################
    #### ratio: %/#bouts per zone
    ########################################################            
    #add percentage in each zone (including 1_Zone instead of perc_1_Zone_while_WG_open, as more comparable to other zone)
    for z in li_Zone:
        df_daily['ratio_percdur_percstays_'+z] = df_daily.apply(lambda x: x['perc_duration_'+z]/x['perc_nbr_stays_'+z] if x['perc_nbr_stays_'+z]!=0 else np.nan, axis=1)    
        
    ########################################################
    #### Nestbox
    ########################################################
    print('----------------NESTBOX')
    ### Time of first staid longer than 15mn (first time with occasion to lay an egg)
    df_init['is_day'] = df_init['Timestamp'].map(lambda x: is_day(x, dico_night_hour))
    #note that minuit is: 0, and its date should be as 1,2 (day-1, day)
    df_init = df_init[df_init['is_day']].copy()
    df_init['level'] = df_init['date'].copy()
    
    #compute duration
    li_df = []
    #more efficient to do it per hen, as it wont need to search in the whole dataframe, and we can simply shift the timestamp column
    for i, df_hen in tqdm.tqdm(df_init.groupby(['HenID'])):
        #as the next record date (sort by date, then simply shift by one row and add nan at then end)
        df_hen = df_hen.sort_values(['Timestamp'], ascending=True)
        #same date, one must take the last recorded one & sorting by date might change it. Also it already should be sorted by date
        df_hen['next_record_date'] = df_hen['Timestamp'].tolist()[1:]+[np.nan]
        #compute duration
        df_hen['duration'] = df_hen.apply(lambda x: (x['next_record_date']-x['Timestamp']).total_seconds(), axis=1)
        li_df.append(df_hen)
    #put again in one dataframe
    df_init = pd.concat(li_df)
    df_init['tuple_Timestamp_duration'] = df_init.apply(lambda x: (x['Timestamp'], x['duration']), axis=1)
    df_zone = df_init.groupby(['HenID','level','Zone'])['tuple_Timestamp_duration'].agg(lambda x: list(x)).reset_index()
    df_zone = df_zone.pivot(values='tuple_Timestamp_duration', index=['HenID', 'level'], columns='Zone').reset_index()
    #df_zone has some nan now due to the pivot
    df_zone.rename(columns={x:str(x)+'_tuple_ts_dur' for x in df['Zone'].unique()}, inplace=True)
    #df_zone['level'] = df_zone['level'].map(lambda x: dt.datetime(x.year,x.month,x.day)) 
    df_daily = pd.merge(df_daily, df_zone, on=['HenID','level'], how='outer')
    df_daily['Nestbox_time_of_first_staid_longer_than'+str(nestbox_sec)+'sec'] = df_daily['4_Zone_tuple_ts_dur'].map(lambda x: min_date_nestbox(x, nestbox_sec))        
    #latency_since15mnnestbox_h == 'h until light turn off after first transition longer than 15mn to nestboxzone
    df_daily['latency_since15mnnestbox_h'] = df_daily['Nestbox_time_of_first_staid_longer_than'+str(nestbox_sec)+'sec'].map(lambda x: (dt.datetime(x.year,x.month,x.day,dico_night_hour[correct_key(x, dico_night_hour)]['end_h'],dico_night_hour[correct_key(x, dico_night_hour)]['end_m'],0)-x).total_seconds()/60/60 if x is not pd.NaT else 0)
    
    ### successful intrusion ratio: (#staid <successfullIntrusionHour h longer than nestbox_sec) / (#of staid <successfullIntrusionHour h)
    df_daily['sucessIntrusion_'+str(successfullIntrusionHour)] = df_daily['4_Zone_tuple_ts_dur'].map(lambda x: successfullIntrusion(successfullIntrusionHour,x,nestbox_sec))
    df_daily['sucessIntrusion_'+str(successfullIntrusionHour)] = df_daily['sucessIntrusion_'+str(successfullIntrusionHour)].fillna(0.0)
    
    ### before 10h (laying egg purpose): number of visits & longest staid
    df_daily['BNestboxHour_nbrd'] = df_daily['4_Zone_tuple_ts_dur'].map(lambda x: nbr_visit_and_longeststay(BNestboxHour, x,'before'))
    df_daily['B'+str(BNestboxHour)+'h_Nestbox_nbrvisit'] = df_daily['BNestboxHour_nbrd'].map(lambda x: x[0])
    df_daily['B'+str(BNestboxHour)+'h_Nestbox_Longestduration'] = df_daily['BNestboxHour_nbrd'].map(lambda x: x[1])
    
    ### after 10h (hiding purpose): number of visits & longest staid
    df_daily['ANestboxHour_nbrd'] = df_daily['4_Zone_tuple_ts_dur'].map(lambda x: nbr_visit_and_longeststay(ANestboxHour, x,'after'))
    df_daily['A'+str(ANestboxHour)+'h_Nestbox_nbrvisit'] = df_daily['ANestboxHour_nbrd'].map(lambda x: x[0])
    df_daily['A'+str(ANestboxHour)+'h_Nestbox_Longestduration'] = df_daily['ANestboxHour_nbrd'].map(lambda x: x[1])

    ### Time of first visit longer 15mn - Time of first visit
    V1 = 'NBtimefirstvisitlonger'+str(nestbox_sec)+'_minus_time1visit'
    V2 = 'Nestbox_time_of_first_staid_longer_than'+str(nestbox_sec)+'sec'
    df_daily[V1] = df_daily.apply(lambda x: (x[V2]-x['FirstTimestamp_4_Zone']).total_seconds() if ((x[V2] is not pd.NaT) &\
                                  (x['FirstTimestamp_4_Zone'] is not pd.NaT))\
                                  else pd.NaT, axis=1)
    #df_daily[[V1,V2,'FirstTimestamp_4_Zone']].head(20)
    df_daily[V1] = df_daily[V1].fillna('no_visit_longer_than_'+str(nestbox_sec))
        
                                                          
    ########################################################
    #### Activity peak
    ########################################################
    #time of the day when the bird did li_perc_activity% of his total transition of the day 
    li_col = [x for x in df_daily.columns if '_tuple_ts_dur' in str(x)]
    df_daily['list_timestamps'] = df_daily.apply(lambda x: [i[0] for l in [x[c] for c in li_col if len(str(x[c]))>3] for i in l],
                                                 axis=1)
    df_daily['list_timestamps_seondsOfTheDay'] = df_daily['list_timestamps'].map(lambda x: [t.hour*60*60+t.minute*60+t.second for t in x])
    #add 5 and 95 if not already here, for the next variable (overall duraiton that hold 5-95% of its transition
    for p in set(li_perc_activity+[5,95]):
        df_daily['activity_'+str(p)+'percentile_h'] = df_daily['list_timestamps_seondsOfTheDay'].map(lambda x: np.nanpercentile(x, 
                                                                                                        p, interpolation='lower')/60/60)
        df_daily['activity_'+str(p)+'percentile_time'] = df_daily['activity_'+str(p)+'percentile_h'].map(lambda x: time.strftime("%H:%M:%S",time.gmtime(x)) if math.isnan(x)==False else np.nan)
        
    #Overall duration that hold 5-95% of its transition    
    #df_daily['duration_5-95percentile_transition'] = df_daily.apply(lambda x: time.strftime("%H:%M:%S",
    #                                                                                        time.gmtime(x['activity_95percentile_h'] -\
    #                                                                                                   x['activity_5percentile_h'])) if\
    #                                                                math.isnan(x['activity_5percentile_h'])==False else np.nan,
    #                                                                axis=1)                                                                 
    #Latest daily transition-first daily transition
    df_daily['duration_last-firsttransition_mn'] = df_daily['list_timestamps_seondsOfTheDay'].map(lambda x: time.gmtime(max(x)-min(x)) if\
                                                                                                  len(x)>0 else np.nan) 
    #convert duration into seconds and put 0 if no mvt: activity is 0
    df_daily['duration_last-firsttransition_mn'] = df_daily['duration_last-firsttransition_mn'].map(lambda x: round(x.tm_hour*60+x.tm_min+x.tm_sec/60,0) if type(x)==time.struct_time else 0) #else np.nan
    
    ########################################################
    #### WG
    ########################################################    
    #went out after its Xmn opening?
    df_daily['in_WG_'+str(WG_after_opening_mn)+'mnAfterOpening'] = df_daily['FirstTimestamp_1_Zone'].map(lambda x: is_ts_before_MN_opening(x, dico_garden_opening_hour, WG_after_opening_mn, date_first_opening_WG, close_dates))
    #if no first entry in wintergarten today, then it did not went out at all
    df_daily['in_WG_'+str(WG_after_opening_mn)+'mnAfterOpening'] = df_daily['in_WG_'+str(WG_after_opening_mn)+'mnAfterOpening'].fillna(False)  
    #max duration in WG
    df_daily['Max_duration_WG_h'] = df_daily['dico_zone_sortedduration'].map(lambda x: max(x.get('1_Zone',[0]))/60/60)

    #add np.nan for the WG var when the WG was close
    #print([x for x in df_daily.columns if '1_Zone' in x]) 
    for v in ['in_WG_'+str(WG_after_opening_mn)+'mnAfterOpening', 'Max_duration_WG_h']+[x for x in df_daily.columns if '1_Zone' in str(x)]:
        df_daily[v] = df_daily.apply(lambda x: x[v] if x['level']>=date_first_opening_WG else np.nan, axis=1)

    ########################################################
    #add basics hens info
    ########################################################
    print('------------ add hen basics info')
    #recompute the df in case thing changed in between
    df_FB_ = FB_process_hen(config)
    li_var_hen = ['HenID','PenID','CLASS','R-Pen','InitialStartDate','early_death','Treatment']
    df_daily = pd.merge(df_daily, df_FB_[li_var_hen], on=['HenID'], how='left')
    #add pen info and match with day info in order to remove the night that we should not use
    #keep all li_var_hen, as some night might exist, while some days not due to disturbances, so we need to add it here
    df_ts_night = pd.merge(df_ts_night, df_FB_[li_var_hen], on=['HenID'], how='left')  
    
    ########################################################
    #remove dates
    ########################################################    
    ######## remove dates linked to specific hens on the day & add tags
    #add the tag separately: per date on the day (to lazy for the night now, and probably no need)
    df_FB_daily = FB_daily(config)
    df_daily['date'] = df_daily['level'].map(lambda x: dt.datetime(x.year,x.month,x.day))
    print(df_daily.shape)
    df_daily = pd.merge(df_daily, df_FB_daily.filter(['HenID','TagID','date','FocalLegringName'], axis=1), on=['date','HenID'],
                        how='inner') 
    print(df_daily.shape)
    df_daily.drop(['date'], axis=1, inplace=True)
    #note that : how=inner in order to oly have records that are correctly associated to a chicken
    #how!= left as we need to remove some records if the system was resetting etc, so we dont want to keep the tracking data of tags 
    #that were not working correctly on that day
    
    ####################################################################################################################################
    ######## remove dates linked to specific system
    print('-------------- Lets remove unwanted dates at PENS level')
    df_daily['date_2remove_penper'] = df_daily.apply(lambda x: x['level'] in df_day[df_day[x['PenID'].split('pen')[-1]+'_Day']==0]['Date'].tolist(), axis=1)
    x0 = df_daily.shape[0]
    df_daily = df_daily[~df_daily['date_2remove_penper']]
    print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))   

    ####################################################################################################################################
    ######## remove dates linked to specific nights and pens
    #hen 129 in pen 8 was on focalbird list but always with non-working days, which will create nan in th pen. So remove it
    df_ts_night = df_ts_night[~df_ts_night['PenID'].isnull()] 
    df_ts_night['night_2remove'] = df_ts_night.apply(lambda x: x['level'] in df_day[df_day[x['PenID'].split('pen')[-1]+'_Night']==0]['Date'].tolist(),axis=1)
    df_ts_night = df_ts_night[~df_ts_night['night_2remove']]
    df_ts_night.drop(['night_2remove'],inplace=True,axis=1)
    #now we can join days and night info: will induce some rows with all the daily var beeing nan, but the night variable having value
    
    try:
        df_ts_night['level'] = df_ts_night['level'].map(lambda x: dt.datetime(x.year,x.month,x.day)) #necessary for the merging
        #ERROR here can happen if initialstartsdate has two value for example!!
        #merge on HenID, level and all other li_var_hen, otherwise we will have duplicated columns
        df_daily = pd.merge(df_daily, df_ts_night, on=li_var_hen+['level'], how='outer')
    except Exception as e: 
        print('ERROOOOOOR',e)
        return [df_ts_night, df_daily, li_var_hen]
    print('All the night variables are: ', df_ts_night.columns)
    print([c for c in df_daily.columns if str(c).startswith('night_level')])
    
    ####################################################################################################################################
    ######## remove dates of tags when they were not giving deviceupdate regularly or before 13.10.2020 as beforehand we couldnt 
    #control for it
    print('-------------- Lets remove dates of tags when they were not giving deviceupdate correctly')
    #verified date: correct
    df_wt = pd.read_csv(os.path.join(path_extracted_data, id_run+'_df_alldeviceinfo.csv'), parse_dates=['date'], dayfirst=True, sep=';') 
    x0 = df_daily.shape[0]
    #keep if it says to be kept in the device file!
    df_daily['date_2keep'] = df_daily.apply(lambda x: x['level'] in df_wt[(df_wt['2keep'])&(df_wt['sender']==x['TagID'])]['date'].tolist(), 
                                                          axis=1)
    x0 = df_daily.shape[0]
    df_daily = df_daily[(df_daily['date_2keep'])|(df_daily['level']<=dt.datetime(2020,10,12))]
    print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))  
    
    ###################################################################################################################################    
    ######## remove all the 30.09.2020 for the tags that still had no transition from before the light went on 
    print('-------------- Lets remove the 30.09.2020 for the tags that still had no transition from before the light went on ')
    x0 = df_daily.shape[0]
    df_daily = df_daily[~((df_daily['level']==dt.datetime(2020,9,30))&(df_daily['verification_daily_total_duration']!=28800))]
    df_daily = df_daily[df_daily['level']>dt.datetime(2020,9,29)]
    print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))
    
    ####################################################################################################################################
    ######## more generally remove all (dates,tagid) with not all seconds tracked (e.g. first day an animal is tracked)
    print('-------------- Lets remove all (dates,tagid) with not all seconds tracked (e.g. first day an animal is tracked), and no night variable')
    df_daily['nbr_h_per_day'] = df_daily['level'].map(lambda x: dico_night_hour[correct_key(x,dico_night_hour)]['nbr_hour'])
    df_daily['correct_amount_of_hour'] = df_daily.apply(lambda x: x['verification_daily_total_nbr_hour']==x['nbr_h_per_day'], axis=1)
    x0 = df_daily.shape[0]
    #as we wont really use the night, lets just remove all the time that corret_maounf of our is incorrect, altought it might remove some correct nights (but otherwise we should keep nights in a separate table
    df_daily = df_daily[df_daily['correct_amount_of_hour']]
    print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))
    
    ######## remove days with only the night and not the days variables, as due to flickering we wotn really use night
    print('-------------- remove days with only the night and not the days variables, as due to flickering we wotn really use night')
    x0 = df_daily.shape[0]
    df_daily = df_daily[~df_daily['Total_number_transition'].isnull()]
    print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))    
            
    ######## remove all above the last official tracked day
    print('-------------- Lets remove all above the last official tracked day')
    x0 = df_daily.shape[0]
    df_daily = df_daily[df_daily['level']<=config.date_max]
    print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))    
    
    
    ########################################################
    #add Device variables: DATA could not be validated, so we remove it
    ########################################################    
    #note that LF counter is basically used to detect problematic tags during daily checks, we wont use it here
    #add movementCounter variable to have an idea on the horizontal movement of the chicken
    #df_deviceVar = pd.read_csv(os.path.join(config.path_extracted_data, config.id_run+'_DeviceVariables.csv'), 
    #                           parse_dates=['level'], dayfirst=True, sep=';', index_col=0) #verified date: correct
    #df_deviceVar['TagID'] = df_deviceVar['TagID'].map(lambda x: 'tag_'+str(int(x)))    
    #df_deviceVar['level'] = df_deviceVar['level'].map(lambda x: dt.datetime(x.year,x.month,x.day))
    #df_deviceVar.drop('night_level',inplace=True, axis=1)
    #df_daily = pd.merge(df_daily, df_deviceVar, on=['TagID', 'level'], how='left')
    
    ########################################################
    #add DOA; DIB
    ########################################################    
    df_daily['DOA'] = df_daily['level'].map(lambda x: (x-config.birth_date).days) 
    df_daily['WOA'] = df_daily['DOA'].map(lambda x: math.ceil(x/7))
    df_daily['DIB'] = df_daily['DOA'].map(lambda x: x-118)
    df_daily['WIB'] = df_daily['DIB'].map(lambda x:  math.ceil(x/7))
    
    #save
    if save:
        print('save')
        df_daily.drop(['zone_list','date_2remove_penper','date_2keep'],inplace=True,axis=1) #verification_daily_total_duration
        df_daily.to_csv(os.path.join(path_extracted_data, id_run+'_daily_'+'_'+str(name_)+'_variables.csv'), sep=';', index=False)

    END_TIME = time.perf_counter()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))
    
    return df_daily

def HenDailyVariable_Origins_simplest(df, config, name_='', timestamp_name='Timestamp', save=True, has_cons_equal_zone=True): 
    
    ''' 
    Note: work with ts that have nan (typically at begining)
    
    Input:
    df_ts: Each row correspond to a specific timestamp, each column to a specific hen timeseries (which column name must start 
        with hen_ ). Must also have a Timestamp and a level column, which will be used to aggregate info and compute variables on these 
        aggregated info
    config: file with parameter
    has_cons_equal_zone: if the initial data has some consecutives euqal zone for hte same hen (that are not necessarily at the same time)
    
    Output:
    Dataframe with according variables'''
    
    #start recording the time it last
    START_TIME = time.perf_counter()
    
    #remove milliseconds now that we cleaned the data (i.e. the records with less than 1seconds duration
    #sort by timestamp
    df.sort_values([timestamp_name], inplace=True)
    #use up to the second level only
    df[timestamp_name] = df[timestamp_name].map(lambda x: dt.datetime(x.year,x.month,x.day,x.hour,x.minute, x.second))
    #remove the first record
    df = df.drop_duplicates(subset=['HenID',timestamp_name], keep='last')
    
    #remove duration if existing in the dataframe to avoid error
    if 'duration' in df.columns:
        df.drop('duration', axis=1, inplace=True)
    df_init = df.copy()
    print('----------------- Create time serie')
    df_ts = time_series_henColumn_tsRow(df, config, col_ts='Zone', ts_with_all_hen_value=False, save=False, hen_time_series=False)
    
    #compute nbr_sec computation here (list of difference between each timestamp, and must always be the same)
    li_ts = df_ts[timestamp_name].tolist()
    li_diff_ts = list(set(list(map(operator.sub, li_ts[1:], li_ts[0:-1]))))
    if len(li_diff_ts)!=1:
        print('ERROR: your timestamp columns have different one to one difference: ', li_diff_ts)
        sys.exit()
    nbr_sec = li_diff_ts[0].total_seconds()
    print('your time series has %d seconds between two timestamps'%nbr_sec)    
    
    ############ initialise parameters from config file
    path_extracted_data = config.path_extracted_data
    id_run = config.id_run
    date_max = config.date_max
    dico_night_hour = config.dico_night_hour
    dico_zone_order = config.dico_zone_order
    date_first_opening_WG = config.date_first_opening_WG
    close_dates = config.close_dates
    dico_garden_opening_hour = config.dico_garden_opening_hour
    path_FocalBird = config.path_FocalBird
    path_Days = config.path_Days
    nestbox_sec = config.nestbox_sec
    ANestboxHour = config.ANestboxHour
    BNestboxHour = config.BNestboxHour
    successfullIntrusionHour = config.successfullIntrusionHour
    WG_after_opening_mn = config.WG_after_opening_mn
    NonTrans_dur_sec = config.NonTrans_dur_sec
    nbr_sec_chaoticmvt_notmiddle = config.nbr_sec_chaoticmvt_notmiddle
    li_nbr_sec_chaoticmvt_middle = config.li_nbr_sec_chaoticmvt_middle
    li_perc_activity = config.li_perc_activity
    #EntropyTimeComputation = config.EntropyTimeComputation
    #NbrData = config.NbrData
    lf_counter = config.lf_counter
    
    ############ small verifications
    #verify columns name of df_ts and select the column we need
    li_hen = [i for i in list(df_ts) if i.startswith('hen_')]
    #verify that the timestamp has same difference than the suggested nbr_sec parameter
    df_ts = df_ts.sort_values(timestamp_name)
    if (df_ts[timestamp_name].iloc[1]-df_ts[timestamp_name].iloc[0]).seconds!=nbr_sec:
        print('ERROR: your timestamp difference does not equal your nbr_sec parameter')
        sys.exit()
        
    ############ separate day and night
    df_ts['is_day'] = df_ts[timestamp_name].map(lambda x: is_day(x, dico_night_hour))
    #night
    df_ts_night = df_ts[~df_ts['is_day']].copy()
    df_ts_night['night_level'] = df_ts_night[timestamp_name].map(lambda x: str(x)[0:-9]+'_'+str(x+dt.timedelta(days=1))[8:10] if\
                                            name_level(x,dico_night_hour) else str(x-dt.timedelta(days=1))[0:-9]+'_'+str(x)[8:10])
    #days
    #note that minuit is: 0, and its date should be as 1,2 (day-1, day)
    df_ts = df_ts[df_ts['is_day']].copy()
    df_ts['level'] = df_ts['date'].copy()
    
    ############ open days to be removed
    df_day = pd.read_csv(path_Days, sep=',', parse_dates=['Date'], dayfirst=True) #date verified: correct

    
    ########################################################
    #### night info 
    ########################################################    
    #have to be done at begining to free memory space
    print('----------------- main night zone and nbr of transitions over night....')
    df_ts_night = pd.melt(df_ts_night.filter([timestamp_name,'night_level']+li_hen), id_vars=[timestamp_name,'night_level'],
                          value_vars=li_hen)
    df_ts_night.rename(columns={'variable':'HenID','value':'Zone'}, inplace=True)
    df_ts_night = df_ts_night[~df_ts_night['Zone'].isnull()].groupby(['HenID','night_level']).agg(
                            night_Max_duration_zones=pd.NamedAgg(column='Zone', aggfunc=lambda x: max_duration_zones(x)),
                            night_duration_Z5=pd.NamedAgg(column='Zone', aggfunc=lambda x: duration_Z5(x)),
                            night_Total_number_transition=pd.NamedAgg(column='Zone', 
                                                                      aggfunc=lambda x: nbr_transition(list((x))))).reset_index()
    df_ts_night['is_mvt_night'] = df_ts_night['night_Total_number_transition'].map(lambda x: int(x>0))
    
    #amount of transition per hour during the night
    df_n = df_init.copy()
    df_n['is_day'] = df_n['Timestamp'].map(lambda x: is_day(x, dico_night_hour))
    df_n['night_level'] = df_n['Timestamp'].map(lambda x: str(x)[0:-9]+'_'+str(x+dt.timedelta(days=1))[8:10] if\
                                            name_level(x,dico_night_hour) else str(x-dt.timedelta(days=1))[0:-9]+'_'+str(x)[8:10])
    df_n['hour'] = df_n['Timestamp'].map(lambda x: x.hour)
        
    df_ts_night = df_ts_night.fillna(0) #add 0 to all the one that had no transition over night
    df_ts_night['level'] = df_ts_night['night_level'].map(lambda x: dt.datetime.strptime(x.split('_')[0], '%Y-%m-%d'))
    
    ########################################################    
    ############ one row per unique hen-timestamp 
    ########################################################    
    df = df_ts.filter([timestamp_name,'level']+li_hen).copy()  
    #list of involved level
    li_day = set(df['level'].tolist())  
    df = pd.melt(df, id_vars=[timestamp_name,'level'], value_vars=li_hen)
    df.rename(columns={'variable':'HenID','value':'Zone'}, inplace=True)
    #we define the duration of each row to be the nbr_sec, its better than computing with the next timestamp as if we removed some days
    #due to health-assessemnt, then it will induce wrong durations! also more efficient that way. BUT its an assumption, that the row must
    #be equally spaced and nbr_sec is the duration in between each timestamp
    df['duration_sec'] = nbr_sec
    #list of not nan Zones
    li_Zone = [x for x in df[~df['Zone'].isnull()]['Zone'].unique()]

    ########################################################
    print('----------------- total & percentage duration per Zone in seconds....')
    #one row per day, hen, existingzone
    df_ = df.groupby(['HenID','level','Zone'])['duration_sec'].agg(lambda x: sum(x)).reset_index()
    #one row per day and hen, each columns account for a zone_duration
    df_daily = df_.pivot_table(values='duration_sec', index=['HenID', 'level'], columns='Zone')
    df_daily.rename(columns={x:'duration_'+x for x in li_Zone}, inplace=True)
    #lets verify with total duration
    df_daily['verification_daily_total_duration'] = df_daily.apply(lambda x: np.nansum([x[i] for i in ['duration_'+x for x in li_Zone]]),
                                                                   axis=1)
    df_daily['verification_daily_total_nbr_hour'] = df_daily['verification_daily_total_duration'].map(lambda x: x/60/60)

    df_daily = df_daily.reset_index()
    #replace np.nan duration by 0
    df_daily.replace(np.nan,0, inplace=True)
    #print('The number of hours per \"level\" period is of:')
    #display(df_daily.groupby(['verification_daily_total_nbr_hour'])['level','HenID'].agg(lambda x: list(x)).reset_index())

    #create an ordered list of the normalized duration per zone for chi2distance later (hen will first be sorted by entropy, and 
    #hence we will do this at the end)
    li_zone_dur = [c for c in df_daily.columns if c.startswith('duration_')] #keep same order
    df_daily['dur_values'] = df_daily.apply(lambda x: str([x[i] for i in li_zone_dur]), axis=1)
    df_daily['dur_values'] = df_daily['dur_values'].map(lambda x: eval(x))
    df_daily['dur_values_normalized'] = df_daily['dur_values'].map(lambda x: [i/float(np.sum(x)) if float(np.sum(x))!=0 else 0 for i in x])

    #add percentage in each zone also in zone 1
    for c in ['duration_'+x for x in li_Zone]:
        df_daily['perc_'+c] = df_daily.apply(lambda x: round(x[c]/x['verification_daily_total_duration']*100), axis=1)
    #(Total time outside)/(total time wg is open)
    df_daily['time_wg_open_sec'] = df_daily['level'].map(lambda x: WG_open_time(config,x)) #duratio of WG opening
    df_daily['duration_sincedaystarted_beforeWGopened_sec'] = df_daily['level'].map(lambda x: WG_open_dayduration_untilopen(config,x))
    df_daily['perc_1_Zone_while_WG_open'] = df_daily.apply(lambda x: x['duration_1_Zone']/x['time_wg_open_sec']*100, axis=1)

    ########################################################
    print('----------------- first timestamp in each zone per day & latency var....')
    #why: will be usefull to produce other variables, to verify the code and to use it for some zones
    df_ = df.groupby(['HenID', 'level','Zone'])[timestamp_name].agg(lambda x: min(list(x))).reset_index()
    #agg function = 'first' ats its string value, and the default function is the mean. Here by construction df_ has unique such 
    #values
    df__ = df_.pivot_table(values=timestamp_name, index=['HenID', 'level'], columns='Zone', aggfunc='first')
    df__.rename(columns={x:'FirstTimestamp_'+x for x in li_Zone}, inplace=True)
    df__ = df__.reset_index()
    df_daily = pd.merge(df_daily, df__, how='outer', on=['HenID','level'])
    
    #latency in WG
    #how long was WG open for today
    df_daily['nbr_h_WGopen'] = df_daily['level'].map(lambda x: nbrh_WG_open(x, config))
    df_daily['FirstTimestamp_1_Zone_sec'] = df_daily['FirstTimestamp_1_Zone'].map(lambda x: (x.hour*60*60+x.minute*60+x.second))
    df_daily['latency_1_Zone_h'] = df_daily.apply(lambda x: (x['FirstTimestamp_1_Zone_sec']-
                                                            x['duration_sincedaystarted_beforeWGopened_sec'])/60/60 if x is not pd.NaT else 0, axis=1)    
    
    #when hen never went outside then put the total amount of time wg was open for. later on nan will be put when wg was close
    df_daily['latency_1_Zone_h'] = np.where(df_daily['FirstTimestamp_1_Zone_sec'].isnull(), df_daily['nbr_h_WGopen'],
                                            df_daily['latency_1_Zone_h'])

    ########################################################
    print('----------------- number of Zone (excluding nan)....')
    df_ = df[~df['Zone'].isnull()].groupby(['HenID','level'])['Zone'].agg(lambda x: len(set((x)))).reset_index()
    df_.rename(columns={'Zone':'Total_number_zone'}, inplace=True)
    df_daily = pd.merge(df_daily, df_, how='outer', on=['HenID','level'])
    
        
    ########################################################        
    #compute some variables 
    ########################################################
    #based on a list of zones over a day, where each zone count for the same nbr_sec second
    #e.g.[einstreu,eintreu,rampe,rampe.....]
    #excluding empty zones, because it influences for exemple the entropy computation (if full of nan, then might be more predictable)    
    print('----------------- compute some variables based on a list of zones over a day....')
    #note that entropy gets as input a list of proba distribution (or count of each element)                    
    df_ = df[~df['Zone'].isnull()].groupby(['HenID','level']).agg(
           list_of_durations=pd.NamedAgg(column='Zone', aggfunc=lambda x: list_of_durations(x, nbr_sec)),
           food_related_behavior=pd.NamedAgg(column='Zone', aggfunc=lambda x: food_related_behavior(li_Z=list(x), config=config)),
           food_related_behavior_rp=pd.NamedAgg(column='Zone', aggfunc=lambda x: food_related_behavior(li_Z=list(x), config=config, 
                                                                                                       fake_sec=20*60)),
           food_related_behavior_rm=pd.NamedAgg(column='Zone', aggfunc=lambda x: food_related_behavior(li_Z=list(x), config=config,
                                                                                                      fake_sec=-20*60)),
           nestboxes_related_behavior=pd.NamedAgg(column='Zone', aggfunc=lambda x: nestboxes_related_behavior(li=list(x), config=config)),
           zone_list=pd.NamedAgg(column='Zone', aggfunc=lambda x: tuple(x)),
           list_of_zones=pd.NamedAgg(column='Zone', aggfunc=lambda x: list_of_zones(list(x))), 
           list_Z4=pd.NamedAgg(column='Zone', aggfunc=lambda x: list_Z4(list(x))),
           mid_cum_Z4_sec=pd.NamedAgg(column='Zone', aggfunc=lambda x: mid_cum_Z4_sec(list(x))),
           Total_number_transition=pd.NamedAgg(column='Zone', aggfunc=lambda x: nbr_transition(list(x))),
           nbr_stays=pd.NamedAgg(column='Zone', aggfunc=lambda x: nbr_bouts_per_zone(list(x))),
           distribution_entropy=pd.NamedAgg(column='Zone', aggfunc=lambda x: entropy(list(Counter(list(x)).values()),base=2)), 
           vertical_travel_distance=pd.NamedAgg(column='Zone', aggfunc=lambda x: vertical_travel_distance(list(x))),
           ).reset_index()
    #distribution entropy note: order does not matter, number of 0min in a zone does not matter. compute here to then be able to verify with dur_values, isntead of computing directly on dur_values
    
    df_daily = pd.merge(df_daily, df_, how='outer', on=['HenID','level'])
  
    #nestboxes_related_behavior reliable only once the barn schedule is stable!
    df_daily.loc[df_daily['level']<dt.datetime(2020,11,13,0,0,0),'nestboxes_related_behavior'] = np.nan
    #if spent less than 15minutes in total in the nestbox zone then put 0
    df_daily.loc[df_daily['duration_4_Zone']<15*60,'nestboxes_related_behavior'] = 0

    #total number of stay
    df_daily['nbr_stays_total'] = df_daily['nbr_stays'].map(lambda x: np.nansum(list(x.values())))
    #retrieve info per zone
    for z in li_Zone:
        df_daily['nbr_stays_'+z] = df_daily['nbr_stays'].map(lambda x: x.get(z,0))
    #total number of stay %of stays per zone
    df_daily['nbr_stays_total'] = df_daily['nbr_stays'].map(lambda x: np.nansum(list(x.values())))
    
    ########################################################
    #### WG
    ########################################################    
    print('------------ add WG info')
    #put np.nan when nbr_h_WGopen to all WG var
    for x_ in [x for x in df_daily.columns if '1_Zone' in x]:
        df_daily.loc[df_daily['nbr_h_WGopen']==0, x_] = np.nan
    
    ########################################################
    #add basics hens info
    ########################################################
    print('------------ add hen basics info')
    #recompute the df in case thing changed in between
    df_FB_ = FB_process_hen(config)
    li_var_hen = ['HenID','PenID','CLASS','R-Pen','InitialStartDate','early_death','Treatment']
    df_daily = pd.merge(df_daily, df_FB_[li_var_hen], on=['HenID'], how='left')
    #add pen info and match with day info in order to remove the night that we should not use
    #keep all li_var_hen, as some night might exist, while some days not due to disturbances, so we need to add it here
    df_ts_night = pd.merge(df_ts_night, df_FB_[li_var_hen], on=['HenID'], how='left')  
    
    ########################################################
    #remove dates
    ########################################################    
    ######## remove dates linked to specific hens on the day & add tags
    #add the tag separately: per date on the day (to lazy for the night now, and probably no need)
    df_FB_daily = FB_daily(config)
    df_daily['date'] = df_daily['level'].map(lambda x: dt.datetime(x.year,x.month,x.day))
    print(df_daily.shape)
    df_daily = pd.merge(df_daily, df_FB_daily.filter(['HenID','TagID','date','FocalLegringName'], axis=1), on=['date','HenID'],
                        how='inner') 
    print(df_daily.shape)
    df_daily.drop(['date'], axis=1, inplace=True)
    #note that : how=inner in order to oly have records that are correctly associated to a chicken
    #how!= left as we need to remove some records if the system was resetting etc, so we dont want to keep the tracking data of tags 
    #that were not working correctly on that day

    ####################################################################################################################################
    ######## remove dates linked to specific system
    print('-------------- Lets remove unwanted dates at PENS level')
    df_daily['date_2remove_penper'] = df_daily.apply(lambda x: x['level'] in df_day[df_day[str(x['PenID'])+'_Day']==0]['Date'].tolist(), axis=1)
    x0 = df_daily.shape[0]
    df_daily = df_daily[~df_daily['date_2remove_penper']]
    print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))   

    ####################################################################################################################################
    ######## remove dates linked to specific nights and pens
    #hen 129 in pen 8 was on focalbird list but always with non-working days, which will create nan in th pen. So remove it
    df_ts_night = df_ts_night[~df_ts_night['PenID'].isnull()] 
    df_ts_night['night_2remove'] = df_ts_night.apply(lambda x: x['level'] in df_day[df_day[str(x['PenID'])+'_Night']==0]['Date'].tolist(),axis=1)
    df_ts_night = df_ts_night[~df_ts_night['night_2remove']]
    df_ts_night.drop(['night_2remove'],inplace=True,axis=1)
    #now we can join days and night info: will induce some rows with all the daily var beeing nan, but the night variable having value

    try:
        df_ts_night['level'] = df_ts_night['level'].map(lambda x: dt.datetime(x.year,x.month,x.day)) #necessary for the merging
        #ERROR here can happen if initialstartsdate has two value for example!!
        #merge on HenID, level and all other li_var_hen, otherwise we will have duplicated columns
        df_daily = pd.merge(df_daily, df_ts_night, on=li_var_hen+['level'], how='outer')
    except Exception as e: 
        print('ERROOOOOOR',e)
        return [df_ts_night, df_daily, li_var_hen]
    print('All the night variables are: ', df_ts_night.columns)
    print([c for c in df_daily.columns if str(c).startswith('night_level')])

    ####################################################################################################################################
    ######## remove dates of tags when they were not giving deviceupdate regularly or before 13.10.2020 as beforehand we couldnt 
    #control for it
    print('-------------- Lets remove dates of tags when they were not giving deviceupdate correctly')
    #verified date: correct
    df_wt = pd.read_csv(os.path.join(path_extracted_data, id_run+'_df_alldeviceinfo.csv'), parse_dates=['date'], dayfirst=True, sep=';') 
    x0 = df_daily.shape[0]
    #keep if it says to be kept in the device file!
    df_daily['date_2keep'] = df_daily.apply(lambda x: x['level'] in df_wt[(df_wt['2keep'])&(df_wt['sender']==x['TagID'])]['date'].tolist(), 
                                                          axis=1)
    x0 = df_daily.shape[0]
    df_daily = df_daily[(df_daily['date_2keep'])|(df_daily['level']<=dt.datetime(2020,10,12))]
    print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))  

    ###################################################################################################################################    
    ######## remove all the 30.09.2020 for the tags that still had no transition from before the light went on 
    print('-------------- Lets remove the 30.09.2020 for the tags that still had no transition from before the light went on ')
    x0 = df_daily.shape[0]
    df_daily = df_daily[~((df_daily['level']==dt.datetime(2020,9,30))&(df_daily['verification_daily_total_duration']!=28800))]
    df_daily = df_daily[df_daily['level']>dt.datetime(2020,9,29)]
    print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))

    ####################################################################################################################################
    ######## more generally remove all (dates,tagid) with not all seconds tracked (e.g. first day an animal is tracked)
    print('-------------- Lets remove all (dates,tagid) with not all seconds tracked (e.g. first day an animal is tracked), and no night variable')
    df_daily['nbr_h_per_day'] = df_daily['level'].map(lambda x: dico_night_hour[correct_key(x,dico_night_hour)]['nbr_hour'])
    df_daily['correct_amount_of_hour'] = df_daily.apply(lambda x: x['verification_daily_total_nbr_hour']==x['nbr_h_per_day'], axis=1)
    x0 = df_daily.shape[0]
    #as we wont really use the night, lets just remove all the time that corret_maounf of our is incorrect, altought it might remove some correct nights (but otherwise we should keep nights in a separate table
    df_daily = df_daily[df_daily['correct_amount_of_hour']]
    print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))

    ######## remove days with only the night and not the days variables, as due to flickering we wotn really use night
    print('-------------- remove days with only the night and not the days variables, as due to flickering we wotn really use night')
    x0 = df_daily.shape[0]
    df_daily = df_daily[~df_daily['Total_number_transition'].isnull()]
    print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))    

    ######## remove all above the last official tracked day
    print('-------------- Lets remove all above the last official tracked day')
    x0 = df_daily.shape[0]
    df_daily = df_daily[df_daily['level']<=config.date_max]
    print_color((('By removing the unwanted days we passed from %d to %d timestamp (losing '%(x0,
                df_daily.shape[0]),'black'), (x0-df_daily.shape[0],'red'),(' timestamp)','black')))    

    ########################################################
    #add DOA; DIB
    ########################################################    
    df_daily['DOA'] = df_daily['level'].map(lambda x: (x-config.birth_date).days) 
    df_daily['WOA'] = df_daily['DOA'].map(lambda x: math.ceil(x/7))
    df_daily['DIB'] = df_daily['DOA'].map(lambda x: x-118)
    df_daily['WIB'] = df_daily['DIB'].map(lambda x:  math.ceil(x/7))

    #save
    if save:
        print('save')
        df_daily.drop(['zone_list','date_2remove_penper','date_2keep'],inplace=True,axis=1) #verification_daily_total_duration
        df_daily.to_csv(os.path.join(path_extracted_data, id_run+'_daily_'+'_'+str(name_)+'_variables.csv'), sep=';', index=False)

    END_TIME = time.perf_counter()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))

    return df_daily


def HenEntropy(df_ts, config, ts_name, name_='', timestamp_name='Timestamp'):
    ''' 
    Input:
    df_ts: Each row correspond to a specific timestamp, each column to a specific hen timeseries (which column name must start 
        with hen_ ). Must also have a Timestamp and a level column, which will be used to aggregate info and compute variables on these 
        aggregated info
    config: file with parameter
    
    Output:
    daily dataframe (where daily is according to the level variable) with according variables'''
    

    ############ initialise parameters 
    START_TIME = time.perf_counter() #start recording the time it last
    path_extracted_data = config.path_extracted_data
    id_run = config.id_run
    dico_night_hour = config.dico_night_hour
    dico_zone_order = config.dico_zone_order
    nbr_sec_bining = config.nbr_sec_bining
    
    ############ add correct 'level' variable (i.e. consecutive time slot for night time series)
    df_ts['day'] = df_ts[timestamp_name].map(lambda x: dt.datetime(x.year,x.month,x.day)) 
    df_ts['is_day'] = df_ts[timestamp_name].map(lambda x: is_day(x, dico_night_hour))
    #note that minuit is: 0, and its date should be as 1,2 (day-1, day)
    if ts_name == 'time_serie_night':
        df_ts = df_ts[~df_ts['is_day']].copy()
        df_ts['level'] = df_ts[timestamp_name].map(lambda x: str(x)[0:-9]+'_'+str(x+dt.timedelta(days=1))[8:10] if\
                                                name_level(x,dico_night_hour) else str(x-dt.timedelta(days=1))[0:-9]+'_'+str(x)[8:10])
    elif ts_name == 'time_serie_day':
        df_ts = df_ts[df_ts['is_day']].copy()
        df_ts['level'] = df_ts['day'].copy()
    else:
        print('ERROR: ts_name parameter must either be time_serie_night or time_serie_day')
        sys.exit()
        
    ############ verifications
    #compute nbr_sec computation here (list of difference between each timestamp, and must always be the same)
    li_ts = df_ts[timestamp_name].tolist()
    li_diff_ts = list(set(list(map(operator.sub, li_ts[1:], li_ts[0:-1]))))
    if len(li_diff_ts)!=1:
        print('WARNING: your timestamp columns have different one to one difference: ', li_diff_ts)
        #return(df_ts)
        #sys.exit()
    #verify columns name of df_ts and select the column we need
    li_hen = [i for i in list(df_ts) if i.startswith('hen_')]
    if not all([i in df_ts.columns for i in [timestamp_name,'level']]):
        print('ERROR: your df_ts must have timestamp and level column name')
        sys.exit()
    df = df_ts.filter([timestamp_name,'level']+li_hen).copy()

    ############ one row per unique hen-timestamp 
    df = pd.melt(df, id_vars=[timestamp_name,'level'], value_vars=li_hen)
    df.rename(columns={'variable':'HenID','value':'Zone'}, inplace=True)
    df['duration_sec'] = nbr_sec_bining
    #list of not nan Zones
    li_Zone = [x for x in df[~df['Zone'].isnull()]['Zone'].unique()]

    ########################################################
    print('----------------- total duration per Zone in seconds....')
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
    #print('The number of hours per \"level\" period is of:')
    #display(df_daily.groupby(['verification_daily_total_nbr_hour'])['level','HenID'].agg(lambda x: list(x)).reset_index())
    
    ########################################################
    print('----------------- first time stamp in each zone per day....')
    df_ = df.groupby(['HenID', 'level','Zone'])[timestamp_name].agg(lambda x: min(list(x))).reset_index()
    #agg function = 'first' ats its string value, and the default function is the mean. Here by construction df_ has unique such 
    #values
    df__ = df_.pivot_table(values=timestamp_name, index=['HenID', 'level'], columns='Zone', aggfunc='first')
    df__.rename(columns={x:'FirstTimestamp_'+x for x in li_Zone}, inplace=True)
    df__ = df__.reset_index()
    df_daily = pd.merge(df_daily, df__, how='outer', on=['HenID','level'])
    
    ########################################################
    ####running SampEnt, DistrEnt computed over the whole period and not only the day. creating only the value at 17h (end of the day)
    display(df.head(3))
    print('----------------- Running entropies at end of each level....')
    dico_HenID_day_ent = {}
    for k, df_hen in df.groupby(['HenID']):
        df_hen = df_hen[~df_hen['Zone'].isnull()]
        df_hen = df_hen.sort_values('level',ascending=True) #make sure its sorted
        df_hen['Zone'] = df_hen['Zone'].map(lambda x: dico_zone_order[x])
        dico_HenID_day_ent[k] = {}
        for L in df_hen['level'].unique():
            df_ = df_hen[df_hen['level']<=L]
            ts_value = df_.tail(1)[timestamp_name].values[0]
            li_zone = df_['Zone'].tolist()
            nbr_value = len(li_zone)
            dico_HenID_day_ent[k][pd.to_datetime(L)] = {'SampEnt': sample_entropy(li_zone, order=2, metric='chebyshev'),
                                                        'DistEnt': DistributionEntropy(li_zone), 
                                                        'ts_value': ts_value, 'nbr_value': nbr_value}
    df_daily['RunSampEnt_onLastTsOfEachLevel'] = df_daily.apply(lambda x: dico_HenID_day_ent[x['HenID']][x['level']]['SampEnt'],
                                                                axis=1)
    df_daily['RunDistEnt_onLastTsOfEachLevel'] = df_daily.apply(lambda x: dico_HenID_day_ent[x['HenID']][x['level']]['DistEnt'],
                                                            axis=1)
    df_daily['RunEnt_onLastTsOfEachLevel_nbr_value'] = df_daily.apply(lambda x: dico_HenID_day_ent[x['HenID']][x['level']]
                                                                          ['nbr_value'], axis=1)
    df_daily['RunEnt_onLastTsOfEachLevel_ts_value'] = df_daily.apply(lambda x: dico_HenID_day_ent[x['HenID']][x['level']]
                                                                         ['ts_value'], axis=1)    
        
    #save
    df_daily.to_csv(os.path.join(path_extracted_data, id_run+'_'+ts_name+'_'+name_+'_variables.csv'), sep=';', index=False)
    END_TIME = time.perf_counter()
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

    
##########################################################################################################################################
#################################################### reliability and model measures ######################################################
##########################################################################################################################################
    
def ConfMat(li_true, li_pred, labels, path_save, xlabel_):
    cm = confusion_matrix(li_true, li_pred, labels)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels) 
    ax.set_yticklabels([''] + labels)
    plt.xlabel(xlabel_)
    plt.ylabel('Observed')
    plt.savefig(path_save, bbox_inches='tight')
    #plt.show()
    #print("Classification Report")
    #classification_report(li_true, li_pred)   
    
def res_intodico(res):
    dico_res = {'accuracy': res['accuracy'], 'support': res['macro avg']['support'],
     'macroavg_precision':res['macro avg']['precision'], 'macroavg_recall':res['macro avg']['recall'],'macroavg_f1': res['macro avg']['f1-score'], 
     'weightedavg_precision':res['macro avg']['precision'], 'weightedavg_recall':res['macro avg']['recall'],'weightedavg_f1': res['macro avg']['f1-score']}
    for k,v in res.items():
        if k not in ['accuracy', 'macro avg', 'weighted avg']:
            for k2,v2 in v.items():
                dico_res[k+'_'+k2] = v2
    return dico_res


##########################################################################################################################################
################################################################ Video ###################################################################
##########################################################################################################################################
        
    
#from a video save the most dissimilar images and several consecutively
def create_dissimilar_consecutive_frames_3consimg(video_path, video_name, path_save_images, gap, sim_index, 
                                                  image_name_init='', nbr_consec=3, first_number_frames_to_consider=100000,
                                                  video_change_to_file=None, save_img_on_first_it=False):
    '''save the maximum of non-similar nbr_consev images '''
    #initialise video path
    vp = os.path.join(video_path, video_name)
    
    #check if video exists
    if len(glob.glob(vp))!=1:
        print('the video does not exist at your path: %s'%vp)
        sys.exit()

    #read video (create a threaded video stream)
    video = cv2.VideoCapture(vp)
        
    #loop over frames from the video file stream
    k = 0
    id_ = 0
    while True:

        #take frames and check if we have reached the end of the video
        (grabbed, image) = video.read()
        if not grabbed:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        id_ = id_+1
        
        if save_img_on_first_it:
            if id_ == 1:
                imageio.imwrite(os.path.join(path_save_images, 
                             image_name_init+video_name.split('.')[0]+'_'+str(id_)+'_1.jpg'), 
                image)

                for n in range(nbr_consec-1):

                    #take frames, check if we have reached the end of the video, if not put in black and white and update the id_
                    (grabbed, image) = video.read()
                    if not grabbed:
                        break
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    id_ = id_+1

                    imageio.imwrite(os.path.join(path_save_images, 
                                                 image_name_init+video_name.split('.')[0]+'_'+str(id_)+'_'+str(n+2)+'.jpg'), 
                                    image)
        
        if image is not None: 
            
            #if no benchmarking image yet create one
            if k==0:
                #last image for comparaison
                if image is not None:
                    im_compared = image.copy()    
                k = 1

            #see if image should be save, i.e. if enough dissimilar from the last annotated one
            elif k==1:

                #compute similarity between two possible consecutive annotated images
                sim = compare_ssim(im_compared, image, multichannel=False)

                #if not that similar from last annotation image, save with the next 'nbr_consec-1' frames as one image 
                #& updated benchmarking image
                if sim<sim_index:
                    imageio.imwrite(os.path.join(path_save_images, 
                                                 image_name_init+video_name.split('.')[0]+'_'+str(id_)+'_1.jpg'), 
                                    image)
                    
                    for n in range(nbr_consec-1):
                        
                        #take frames, check if we have reached the end of the video, if not put in black and white 
                        #note that we wont update the id_ for image retrieval
                        (grabbed, image) = video.read()
                        if not grabbed:
                            break
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                      
                        imageio.imwrite(os.path.join(path_save_images, 
                                                     image_name_init+video_name.split('.')[0]+'_'+str(id_)+'_'+str(n+2)+'.jpg'), 
                                        image)

   
                    #last image for comparaison
                    if image is not None:
                        im_compared = image.copy() 
                    k = k+1

                #if similar save for detection and continue until find one not similar to save
                else:
                    k = 1
            else:
                k = k+1

            if k%(gap+2)==0:
                k = 1
            
            id_ = id_+1
            #to be verified exactly
            if id_>=first_number_frames_to_consider:
                break

    #close video
    video.release()
    
    #when all is finish put video in the 'done' folder (and remove from the other folder)
    if video_change_to_file is None:
        video_change_to_file = os.path.join(video_path, 'done') 
    #os.rename(vp, os.path.join(video_change_to_file, video_name) )
    shutil.move(vp, os.path.join(video_change_to_file, video_name) )
    
    
#from a video save the most dissimilar images
def create_dissimilar_consecutive_frames(video_path, path_save_images, gap, sim_index, reverse_rgb=False, image_name_init=''):
    
    #initialise video path
    video_name = video_path.split('\\')[-1]
    
    #check if video exists
    if len(glob.glob(video_path))!=1:
        print('the video does not exist at your path: %s'%video_path)
        sys.exit()

    #read video 
    video = cv2.VideoCapture(video_path)
    
    #create directories if not existing to save all images for annotations
    if not os.path.exists(path_save_images):
        os.makedirs(path_save_images)
        
    # loop over frames from the video file stream
    k = 0
    id_ = 0
    while True:

        #take frames and check if we have reached the end of the video
        (grabbed, image) = video.read()
        if not grabbed:
            break

        if reverse_rgb:
            b,g,r = cv2.split(image)           
            image = cv2.merge([r,g,b])
         
        if image is not None: 
            #if no benchmarking image yet create one
            if k==0:
                im_compared = image.copy()
                k = 1

            #see if image should be saved, i.e. if enough dissimilar from the last annotated one
            elif k==1:

                #compute similarity between two possible consecutive annotated images
                sim = compare_ssim(im_compared, image, multichannel=True)

                #if not that similar from last annotation image, save & updated benchmarking image
                if sim<sim_index:
                    im_compared = image.copy()
                    imageio.imwrite(os.path.join(path_save_images, image_name_init+video_name.split('.')[0]+'_'+str(id_)+'.jpg'), image)
                    k = k+1

                #if similar save for detection and continue until find one not similar to save
                else:
                    k = 1
            else:
                k = k+1

            if k%(gap+2)==0:
                k = 1

            #update id of frame
            id_ = id_+1

    #close video
    video.release()
    
    

##########################################################################################################################################
################################################################ visual ##################################################################
##########################################################################################################################################

def ts_visual(df_, dmin, dmax, path_, dico_h_cl, name=''):
    
    '''The dictionary is here to tell which hens to do andwaht is the name of the folder its plot should be saved in
    for more than one days ts'''
    START_TIME = time.perf_counter()
    
    #create one folder per cluster to save
    for clID in set(dico_h_cl.values()):
        path__ = os.path.join(path_, name+str(clID))
        if not os.path.exists(path__):
            os.makedirs(path__)
                
    #select part of interest
    df_ = df_[(df_['date']<=dmax)&(df_['date']>=dmin)].copy()
    print(df_.shape)
    #to sort the yaxis
    dico_zone_order = {'1_Zone':0, '2_Zone':1, '3_Zone':2, '4_Zone':3, '5_Zone':4}
    li_date = df_['date'].unique()
    for henID, df_plt in tqdm.tqdm(df_.groupby(['HenID'])):
        #try:
        df_plt = df_plt.sort_values(['Timestamp']).copy()
        if henID in dico_h_cl:
            clID = dico_h_cl[henID]
            c = 1 ; l = len(li_date)
            fig, ax = plt.subplots(figsize=(c*5, l*1))
            i = 1
            #if len(df_plt['date'].unique())==len(li_date):
            mi = min(df_plt['Timestamp'].tolist()) ; ma = max(df_plt['Timestamp'].tolist())
            Daterange = pd.date_range(start = mi+dt.timedelta(seconds=(60-mi.second)), 
                                      end = ma-dt.timedelta(seconds=(ma.second+1)), 
                                      freq = 'S') 
            df_plt_ = df_plt.copy()
            df_plt_.set_index('Timestamp', inplace=True)
            df_plt_ = df_plt_.reindex(Daterange, method='ffill').reset_index()
            df_plt_.rename(columns={'index':'Timestamp'}, inplace=True)
            #add date again, as the reindexing also extended the date
            df_plt_['date'] = df_plt_['Timestamp'].map(lambda x: dt.datetime.date(x))
            #remove first & last date
            df_plt_ = df_plt_[~df_plt_['date'].isin([max(df_plt_['date'].tolist()),min(df_plt_['date'].tolist())])]
            #put xlabel into numbers for the ploting
            df_plt_['Zone'] = df_plt_['Zone'].map(lambda x: int(dico_zone_order[x]))   
            for d, df_plt__ in df_plt_.groupby(['date']):
                plt.subplot(l,c,i)
                df_plt___ = df_plt__.copy()
                plt.tight_layout(pad=0.3) #add spacing between each plot
                plt.yticks([0,1,2,3,4], ['Winter garden', 'Litter', 'Lower perch','Nestbox','Top floor'])
                i = i+1
                plt.plot(df_plt___['Timestamp'].tolist(), df_plt___['Zone'].tolist(), linewidth=1)
                li_hour = pd.date_range(start = d,  end = d+dt.timedelta(days=1), freq = 'H')
                plt.xticks(li_hour , [str(i.hour)+'h' for i in li_hour], fontsize=5)                
                plt.yticks(fontsize=8)
                plt.xlabel(str(d).split('T')[0], size=7)
                plt.ylim(0, 4.2)  

            #print(henID+'_'+str(dmin).split(' ')[0]+'_'+str(dmax).split(' ')[0]+'.png') #does not work
            name_picture = str(dmin).split(' ')[0].replace('-','')+'_'+str(dmax).split(' ')[0].replace('-','')+'_'+henID+'.png'
            plt.savefig(os.path.join(path_, name+str(clID), name_picture), format='png', bbox_inches='tight', dpi=300)
            plt.clf()
            plt.close("all")
            #except Exception as e:
            #    print('------------')
            #    print('Error with hen: ',henID)
            #    print(e)
            #    pass
    END_TIME = time.perf_counter()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))
    
    
##########################################################################################################################################
################################################################ others ##################################################################
##########################################################################################################################################
    
#all combination of any size
def all_subsets(ss, max_size=None):
    if max_size==None:
        return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))
    else:
        return chain(*map(lambda x: combinations(ss, x), range(0, max_size+1)))
#small example
#li_subset = []
#for subset in all_subsets([1,3,5]):
#    print(subset)
#(), (1,),(3,),(5,),(1, 3),(1, 5),(3, 5),(1, 3, 5)
#for subset in all_subsets([1,3,5], max_size=2):
#    print(subset)
#(), (1,),(3,),(5,),(1, 3),(1, 5),(3, 5)
    