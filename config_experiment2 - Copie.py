import os
import numpy as np
import datetime as dt


#18.08 - 14.07 

#id of run
#this id should be changed each time you want to generate new results (i.e. be saving everyting without deleting what was already saved. Typically when you modify a function in the "utils.py" file and you would like to compare the results
id_run = 'V2_NE_select_' #'v1'


#choose folder names where the initial data and the extracted data/information should be saved. 
#the extracted_path will be created by the computer if it doesnt exist already
#the initial data path should already exist and have the data in the following form:
#path_initial_data\Abteile 3_5\log_*
#path_initial_data\Abteile 10_12\log_*
#path_initial_data = r'D:\vm_exchange\AVIFORUM\data\initial_data_2experiment'
path_initial_data = r'R:\VPHI\Welfare\2- Research Projects\Laura C-Klara G\tracking_Data'
path_extracted_data = r'D:\vm_exchange\AVIFORUM\data\extracted_info_experiment2_VF'


#list of date to be removed due to health assessment or anything else
li_date2remove = list(set([dt.datetime(2019,11,28),dt.datetime(2019,12,9),
                  dt.datetime(2020,1,6), dt.datetime(2020,2,10),dt.datetime(2020,3,16),
                  dt.datetime(2020,5,4),dt.datetime(2020,6,2),dt.datetime(2020,6,29),dt.datetime(2020,6,30),
                  dt.datetime(2019,11,29),dt.datetime(2019,12,2),dt.datetime(2019,12,4),dt.datetime(2019,12,21),
                  dt.datetime(2019,12,22),dt.datetime(2019,12,23),
                  dt.datetime(2019,12,6),dt.datetime(2019,12,11),dt.datetime(2020,1,12), dt.datetime(2020,2,13),dt.datetime(2020,3,17)
                  ]+[dt.datetime(2020,2,2)-dt.timedelta(days=x) for x in range(13)]+\
                  [dt.datetime(2020,1,7)-dt.timedelta(days=x) for x in range(13)]))

#dates to be removed from specifc pens
dico_date2remove_pens = defaultdict(list)


#min and max date to consider, other date will be removed by the "preprocessing_*()" function
date_min = dt.datetime(2020,6,7,0,0,0) #dt.datetime(2019,11,12,0,0,0)
date_max = dt.datetime(2020,6,28,23,59,59) #dt.datetime(2019,11,24,23,59,59)


#pen associated to zone None in the experiment2
dico_zone_matching = None

#bining value
nbr_sec_bining = 60

#list of hours that must be associated to night, and to day, example:
#dico_night_hour = {dt.datetime(2019,11,15,0,0,0):{'start_day_h':3,'start_day_m':5,'end_day_h':17,'end_day_m':25},
#                   dt.datetime(2019,12,15,0,0,0):{'start_day_h':3,'start_day_m':0,'end_day_h':16,'end_day_m':15}}
#means that all record below and including date 2019.11.15 will have a day from 3:05 to 17:25, and all records from
#]2019-11-15 to 2019-12-15] will have a day from 3:00 to 16:15
dico_night_hour = {dt.datetime(2022,12,15,0,0,0):{'start_day_h':2,'start_day_m':0,'end_day_h':17,'end_day_m':0},}


#initial zone matching to general names, will be used in the "general_cleaning()" function
#ATTENTION: first character of the new name should be unique (will be used in flickering situation, to find pattern (e.f. brbrbr) etc)
dico_matching = {'Tier 1':'2 Zone', 
                 'Tier 4 + obere Stange':'5 Zone', 
                 'Tier 2 (mini 10)':'3 Zone', 
                 'Tier 2 + untere Stange':'3 Zone',
                 'Tier 3 Rampe + Nestbox': '4 Zone',
                 'Tier 2 (mini 11)':'3 Zone', 
                 'Wintergarten':'1 Zone', 
                 'Tier 2':'3 Zone', 
                 'Tier 4':'5 Zone',
                 'Tier 2 (mini 4)':'3 Zone', 
                 'Tier 3 - Rampe + Nestbox':'4 Zone', 
                 'Tier 2 (mini 5)':'3 Zone',
                 'Tier 2 (mini 12)':'3 Zone', 
                 'Tier 2 (mini 3)':'3 Zone'}

#should be one of the new name from dico_matching. Used in "verification_based_on_initial_record()" function
outside_zone = '1 Zone'

#SHOULD NOT BE CHANGED!
#the difference between two timestamp of the time series, and hence will be used for the variables computation and the bining
#for now we set it to one (simpler for bining)
#nbr_sec = 1

#define parameters for variables computations
#restrict time series to one value per ValueDelta seconds
ValueDelta = 60
#each EntropyTimeComputation values of ts we compute the variables 
#--> compute entropy each EntropyTimeComputation*ValueDelta seconds
EntropyTimeComputation = 30 
#the shifted variables includes the values of the last NbrData4Shift seconds
#--> each shifted variables will use NbrData/ValueDelta values, while the running will use all previous values starting from 
#the time stamp with at least NbrData/ValueDelta previous values
NbrData = 60*60*2
print('we restrict the time series to one value per %d seconds \nwe compute the complexity variables each %d minutes \
\neach variables includes the values of at least the last %.2f minutes \
(i.e. are using %.2f values)'%(ValueDelta, (EntropyTimeComputation*ValueDelta)/60, NbrData/60, NbrData/ValueDelta))

#------------------------------------------------ CLEANING TO FINISH ---------------------------------------------


#number of seconds that a hens needs to stay at least in a zone to count as a true transition for flickering type 1
#note that if nbr_sec =3, then from 2019-07-07 22:09:32 to 2019-07-07 22:09:35 , there is at least 2 seconds so its consider
#as a transition, but not 32 - 34 
nbr_sec_flickering1 = 3

#maximum number of different zone to be included in a flickering type2 event
nbrZone = 2

#define the impossible mouvement from one zone to another, consider only the first caractere of the zones defined by dico_matching
#each possible value should be put as a value, with empty list if no impossible movement is possible
dico_impossible_mvt_to = {'1':['3','4','5'],
                          '2':['5'],
                          '3':['1'],
                          '4':['1'],
                          '5':['1']}

#set of zone that shoul dnot be identified by flickering 2 (i.e. when they would be, and if thats actually correct sequence, then they
#might be very long sequences like er, which will slow down the process
#if computing the flickering type2 takes too long then you should probably think if you did not miss some possible combination
#leave an empty list if no such case apply to your data
li_not_flickering2 = []

#zone order for visual, we should keep interzone as its stills ome information and also to be sure we always have the same number of hen at each timestamp. All the values should be unique
#ATTENTION: start from 0 
#ATTENTION: all values of dico_matching should have a key in the dico_zone_order 
#ATTENTION: only the zone from this dictionary will be used!!
'''dico_zone_order = {'1 Zone':0,
                   'Interzone_12':1,
                   '2 Zone':2,
                   'Interzone_23':3,
                   '3 Zone':4,
                   'Interzone_34':5,
                   '4 Zone':6,
                   'Interzone_45':7,
                   '5 Zone':8}'''

dico_zone_order = {'1 Zone':0,
                   '2 Zone':1,
                   '3 Zone':2,
                   '4 Zone':3,
                   '5 Zone':4}





