import os
import numpy as np
import datetime as dt
from collections import defaultdict
import pandas as pd

#HA1, HA2, HA3: relocation
#HA4: end of experiment no relocation but health assessments
#8.10.2021: population in barn 4, so we woud use the day after only!

li_initdate = [dt.datetime(2021,10,9), dt.datetime(2021,10,10), dt.datetime(2021,10,11), 
                   dt.datetime(2021,10,16), dt.datetime(2021,10,17), dt.datetime(2021,10,18)]
#health assessment days
li_HA = [dt.datetime(2021,11,30), dt.datetime(2022,2,8), dt.datetime(2022,4,12), dt.datetime(2022,7,5)]
#date_range included both given dates
li_HA1 = list(pd.date_range(start=dt.datetime(2021,11,23), end=dt.datetime(2021,12,1),freq='D'))#tracking before & 1 day after (error)
li_HA2 = list(pd.date_range(start=dt.datetime(2022,2,1), end=dt.datetime(2022,2,16), freq = 'D')) #tracking before & after 
li_HA3 = list(pd.date_range(start=dt.datetime(2022,4,5), end=dt.datetime(2022,4,20), freq = 'D')) #tracking before & after
li_HA4 = list(pd.date_range(start=dt.datetime(2022,6,28), end=dt.datetime(2022,7,4), freq = 'D')) #only tracking before (end experiment)
li_tracking_date = li_initdate + li_HA1 + li_HA2 + li_HA3 + li_HA4
li_tracking_date = [dt.datetime(i.year, i.month, i.day)  for i in li_tracking_date if i not in li_HA]
#ATTENTION: Taking out the manure/shit this week will not happen on Thursday 24.12 but on Wednesday (23.12.)

#id of run
#this id should be changed each time you want to generate new results (i.e. be saving everyting without deleting what was already saved. Typically when you modify a function in the "utils.py" file and you would like to compare the results
id_run = 'EXP2_'

#choose folder names where the initial data and the extracted data/information should be saved. 
#the extracted_path will be created by the computer if it doesnt exist already
#the initial data path should already exist and have the data in the following form:
#path_initial_data\Abteile 3_5\log_*
#path_initial_data\Abteile 10_12\log_*
#CSV
focal_name = 'Focalbirds_12-07-2022.csv'
path_ = r'G:\VPHI\Welfare\2- Research Projects\Camille Montalcini\Origins.GS'
path_initial_data = os.path.join(path_,'GantnerSystem','DATA')
path_dataoutput = os.path.join(path_,'DataOutput')
path_extracted_data = os.path.join(path_dataoutput,'TrackingSystem') 
path_extracted_HA = os.path.join(path_dataoutput,'HA') 

path_FocalBird = os.path.join( path_,'FOCAL BIRDS',focal_name)
path_performance = os.path.join(path_,'Productivity')

#add id_run for readibility
path_extracted_data_daily_check = os.path.join(path_extracted_data, 'DailyVerifications') #not linked to the id_run as its more general
path_extracted_data = os.path.join(path_extracted_data, id_run)
path_extracted_HA_visual = os.path.join(path_extracted_HA,'visual')

#add other usefulle directories
path_extracted_data_SNA = os.path.join(path_extracted_data, 'SNA')

#defines the color palette
#https://matplotlib.org/stable/gallery/color/named_colors.html
#https://matplotlib.org/3.1.1/tutorials/colors/colors.html
pal_ = {'Trackingsystem_Zone':'gold','Binning_Zone':'blue', 'Model_Zone':'deepskyblue','ThresholdOnDuration_Zone':'coral',
       'Unprocessed records':'gold','BIN-dataset':'blue', 'ML-dataset':'deepskyblue','TD-dataset':'coral'}
pal_zone = {'Wintergarden':'green', 'Winter garden':'green', 'Litter':'olive', 'Lower perch':'peru', 
            'Nestbox':'orangered', 'Top floor':'darkred'}
pal_tracking_system = {'TrackingSystem 8-9':'orange', 'TrackingSystem 10-12':'blue', 'TrackingSystem 3-5':'hotpink'}
pal_pens = {'Pen 3':'pink', 'Pen 4':'hotpink', 'Pen 5':'magenta', 
            'Pen 8':'navajowhite', 'Pen 9':'darkorange', 
            'Pen 10':'skyblue', 'Pen 11':'blue', 'Pen 12':'cadetblue'}
dico_pen_ts = {3:'TrackingSystem 3-5',
              4:'TrackingSystem 3-5',
              5:'TrackingSystem 3-5',
              8:'TrackingSystem 8-9',
              9:'TrackingSystem 8-9',
              10:'TrackingSystem 10-12',
              11:'TrackingSystem 10-12',
              12:'TrackingSystem 10-12',
              'Pen 3':'TrackingSystem 3-5',
              'Pen 4':'TrackingSystem 3-5',
              'Pen 5':'TrackingSystem 3-5',
              'Pen 8':'TrackingSystem 8-9',
              'Pen 9':'TrackingSystem 8-9',
              'Pen 10':'TrackingSystem 10-12',
              'Pen 11':'TrackingSystem 10-12',
              'Pen 12':'TrackingSystem 10-12',
              'pen3':'TrackingSystem 3-5',
              'pen4':'TrackingSystem 3-5',
              'pen5':'TrackingSystem 3-5',
              'pen8':'TrackingSystem 8-9',
              'pen9':'TrackingSystem 8-9',
              'pen10':'TrackingSystem 10-12',
              'pen11':'TrackingSystem 10-12',
              'pen12':'TrackingSystem 10-12'}

#Adatability study
#li_binmn = [5,10,15,20,30]
#penalty = 0
#window: we set up so that the animal can have up to 1h of movement wihtout penality
#dico_window = {5:12, 10:6, 15:4, 20:3, 30:2}

#expriment starting date
starting_date = dt.datetime(2021,10,8)

#day of birth to compute day of age
birth_date = dt.datetime(2021,6,9) #DOA 1 = 2020-6-4

#nestbox time:
#Nestbox Time 16:00 close / 02:05 open. 
        
#garden opening hours: until (included) date
#The opening of the wintergarden happens automatically but the closing time is written down as soon as the AKB of the first pen is closed. Normally it takes about 10 minutes to close the rest of the AKBs.
date_first_opening_WG = dt.datetime(2021,11,1,0,0,0)
close_dates = [dt.datetime(2022,1,11,0,0,0), dt.datetime(2022,5,31,0,0,0)]
dico_garden_opening_hour = {dt.datetime(2021, 11, 23,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                             dt.datetime(2021, 11, 24,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                             dt.datetime(2021, 11, 25,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                             dt.datetime(2021, 11, 26,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                             dt.datetime(2021, 11, 27,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                             dt.datetime(2021, 11, 28,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                             dt.datetime(2021, 11, 29,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                             dt.datetime(2021, 12, 1,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                             dt.datetime(2022, 2, 1,0,0,0):{'start_h':11,'start_m':45,'end_h':16,'end_m':20},
                             dt.datetime(2022, 2, 2,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                             dt.datetime(2022, 2, 3,0,0,0):{'start_h':10,'start_m':0,'end_h':17,'end_m':0},
                             dt.datetime(2022, 2, 4,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                             dt.datetime(2022, 2, 5,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                             dt.datetime(2022, 2, 6,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                             dt.datetime(2022, 2, 7,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':25},
                             dt.datetime(2022, 2, 9,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':25},
                             dt.datetime(2022, 2, 10,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                             dt.datetime(2022, 2, 11,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                             dt.datetime(2022, 2, 12,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                             dt.datetime(2022, 2, 13,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                             dt.datetime(2022, 2, 14,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                             dt.datetime(2022, 2, 15,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                             dt.datetime(2022, 2, 16,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                             dt.datetime(2022, 4, 5,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                             dt.datetime(2022, 4, 6,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                             dt.datetime(2022, 4, 7,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                             dt.datetime(2022, 4, 8,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                             dt.datetime(2022, 4, 9,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                             dt.datetime(2022, 4, 10,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                             dt.datetime(2022, 4, 11,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                             dt.datetime(2022, 4, 13,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                             dt.datetime(2022, 4, 14,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                             dt.datetime(2022, 4, 15,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                             dt.datetime(2022, 4, 16,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                             dt.datetime(2022, 4, 17,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                             dt.datetime(2022, 4, 18,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                             dt.datetime(2022, 4, 19,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                             dt.datetime(2022, 4, 20,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                             dt.datetime(2022, 6, 28,0,0,0):{'start_h':8,'start_m':0,'end_h':16,'end_m':20},
                             dt.datetime(2022, 6, 29,0,0,0):{'start_h':8,'start_m':0,'end_h':16,'end_m':20},
                             dt.datetime(2022, 6, 30,0,0,0):{'start_h':8,'start_m':0,'end_h':16,'end_m':20},
                             dt.datetime(2022, 7, 1,0,0,0):{'start_h':8,'start_m':0,'end_h':16,'end_m':20},
                             dt.datetime(2022, 7, 2,0,0,0):{'start_h':8,'start_m':0,'end_h':16,'end_m':10},
                             dt.datetime(2022, 7, 3,0,0,0):{'start_h':8,'start_m':0,'end_h':16,'end_m':10},
                             dt.datetime(2022, 7, 4,0,0,0):{'start_h':8,'start_m':0,'end_h':16,'end_m':30}}
#the keys are included into their values information (i.e. datex:{open time, close time}, date x is opening at that time and closing at that time too (i.e. until (included) date x)

#went out after its WG_after_opening_mn mn opening of WG?
WG_after_opening_mn = 15

#min and max date to consider, all other dates will be removed by the "preprocessing_*()" function
date_min = dt.datetime(2021,10,8,0,0,0) 
date_max = min(dt.datetime.now(), dt.datetime(2022,7,5,23,59,59))


#code associated to each actual readers
dico_rc_sys = {'192.168.1.75':'Reader Pen 3-5',
               '192.168.1.77':'Reader Pen 8-9',
               '192.168.1.78':'Reader Pen 10-12'}

#pen associated to zone None in the experiment2
dico_zone_matching = None

#maximum number of seconds that is allowed to be removed due to the model-cleaning method 
nbr_maxdur2beremoved = 900

#list of hours that must be associated to night, and to day, example:
#dico_night_hour = {dt.datetime(2019,11,15,0,0,0):{'start_day_h':3,'start_day_m':5,'end_day_h':17,'end_day_m':25},
#                   dt.datetime(2019,12,15,0,0,0):{'start_day_h':3,'start_day_m':0,'end_day_h':16,'end_day_m':15}}
#means that all record below and including date 2019.11.15 will have a day from 3:05 to 17:25, and all records from
#]2019-11-15 to 2019-12-15] will have a day from 3:00 to 16:15
dico_night_hour = {dt.datetime(2021,9,30,0,0,0): {'start_h':8,'start_m':0,'end_h':17,'end_m':0,'nbr_hour':9},
                   dt.datetime(2021,10,24,0,0,0): {'start_h':8,'start_m':0,'end_h':17,'end_m':0,'nbr_hour':9},#until 10.24 its 8-17 
                   dt.datetime(2021,10,27,0,0,0): {'start_h':7,'start_m':30,'end_h':17,'end_m':0,'nbr_hour':9.5},
                   dt.datetime(2021,10,28,0,0,0): {'start_h':7,'start_m':0,'end_h':17,'end_m':0,'nbr_hour':10},
                   dt.datetime(2021,11,1,0,0,0): {'start_h':6,'start_m':0,'end_h':17,'end_m':0,'nbr_hour':11},
                   dt.datetime(2021,11,5,0,0,0): {'start_h':5,'start_m':0,'end_h':17,'end_m':0,'nbr_hour':12},
                   dt.datetime(2021,11,13,0,0,0): {'start_h':4,'start_m':30,'end_h':17,'end_m':0,'nbr_hour':12.5},
                   dt.datetime(2021,11,18,0,0,0): {'start_h':4,'start_m':0,'end_h':17,'end_m':0,'nbr_hour':13},
                   dt.datetime(2021,11,25,0,0,0): {'start_h':3,'start_m':30,'end_h':17,'end_m':0,'nbr_hour':13.5},
                   dt.datetime(2022,8,5,0,0,0): {'start_h':3,'start_m':0,'end_h':17,'end_m':0,'nbr_hour':14}}

#nestbox time that it must stay at least to have a chance biologically to be able to lay an egg in the nextbox
nestbox_sec = 15*60

#initial zone matching to general names, will be used in the "general_cleaning()" function
#ATTENTION: first character of the new name should be unique (will be used in flickering situation, to find pattern (e.f. brbrbr) etc)
dico_matching = {'Tier 1':'2_Zone', 
                 'Tier 1 (mini 12)':'2_Zone',
                 'Tier 4 + obere Stange':'5_Zone', 
                 'Tier 2 + untere Stange':'3_Zone',
                 'Tier 3 Rampe + Nestbox': '4_Zone',
                 'Wintergarten':'1_Zone',
                 'Tier 2 (mini 10)':'3_Zone', 
                 'Tier 2 (mini 12)':'3_Zone', 
                 'Tier 2 (mini 9)':'3_Zone', 
                 'Tier 2 (mini 3)':'3_Zone',
                 'Tier 2 (mini 8)':'3_Zone', 
                 'Tier 2 (mini 5)':'3_Zone',
                 'Tier 2 (mini 11)':'3_Zone',
                 'Tier 2 (mini 4)':'3_Zone', 
                 'Tier 2 (mini 12)':'3_Zone',
                 np.nan:np.nan}

#should be one of the new name from dico_matching. Used in "verification_based_on_initial_record()" function
outside_zone = '1 Zone'

dico_zone_order = {'1_Zone':0,
                   '2_Zone':1,
                   '3_Zone':2,
                   '4_Zone':3,
                   '5_Zone':4}
dico_zone_plot_name = {'1_Zone':'Winter garden',
                       '2_Zone':'Litter',
                       '3_Zone':'Lower perch',
                       '4_Zone':'Nestbox',
                       '5_Zone':'Top floor'}


