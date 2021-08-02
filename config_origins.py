import os
import numpy as np
import datetime as dt
from collections import defaultdict


#ATTENTION: Taking out the manure/shit this week will not happen on Thursday 24.12 but on Wednesday (23.12.)

#id of run
#this id should be changed each time you want to generate new results (i.e. be saving everyting without deleting what was already saved. Typically when you modify a function in the "utils.py" file and you would like to compare the results
id_run = 'correctlightschedule_'
#other available: chapter0_final_

#choose folder names where the initial data and the extracted data/information should be saved. 
#the extracted_path will be created by the computer if it doesnt exist already
#the initial data path should already exist and have the data in the following form:
#path_initial_data\Abteile 3_5\log_*
#path_initial_data\Abteile 10_12\log_*
#CSV
focal_name = 'FocalBirdsInfo26-07-2021.csv'
day_name = 'TrackingDaysWhenCanWeUseWhat.csv'
path_initial_data = r'G:\VPHI\Welfare\2- Research Projects\OFHE2.OriginsE2\GantnerSystem\_dailycheckingSystem'
path_dataoutput = r'G:\VPHI\Welfare\2- Research Projects\OFHE2.OriginsE2\DataOutput'
path_extracted_data = os.path.join(path_dataoutput,'TrackingSystem') 
path_extracted_HA = os.path.join(path_dataoutput,'HA') 

path_FocalBird = os.path.join( r'G:\VPHI\Welfare\2- Research Projects\OFHE2.OriginsE2','FOCAL BIRDS',focal_name)
path_Days = os.path.join( r'G:\VPHI\Welfare\2- Research Projects\OFHE2.OriginsE2','FOCAL BIRDS',day_name)
path_performance = os.path.join(r'G:\VPHI\Welfare\2- Research Projects\OFHE2.OriginsE2\Productivity')

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
pal_tracking_system = {'TrackingSystem 8-9':'orange', 'TrackingSystem 10-12':'c', 'TrackingSystem 3-5':'hotpink'}
pal_pens = {'Pen 3':'pink', 'Pen 4':'hotpink', 'Pen 5':'magenta', 
            'Pen 8':'navajowhite', 'Pen 9':'darkorange', 
            'Pen 10':'skyblue', 'Pen 11':'c', 'Pen 12':'cadetblue'}
pal_class_treat = {'TRAN_MEXP':'royalblue','TRAN_LEXP':'darkblue','OFH_MEXP':'orange','OFH_LEXP':'peru'}
pal_treat = {'TRAN':'royalblue','OFH':'orange', 'TRAN-nonfocal':'lightskyblue', 'OFH-nonfocal':'burlywood'}
pal_class = {'MEXP':'slategray','LEXP':'tan'}
pal_interintre_treatment = {'Intra individuals - OFH':'orange', 'Inter individuals - TRAN':'lightskyblue',
                        'Intra individuals - TRAN':'royalblue', 'Inter individuals - OFH':'burlywood'}

dico_pen_tr = {'pen3':'OFH','pen5':'OFH','pen9':'OFH','pen11':'OFH',
               'pen4':'TRAN','pen8':'TRAN','pen10':'TRAN','pen12':'TRAN',
               'pen6':'TRAN-nonfocal', 'pen7':'OFH-nonfocal'}
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
li_binmn = [5,10,15,20,30]
penalty = 0
#window: we set up so that the animal can have up to 1h of movement wihtout penality
dico_window = {5:12, 10:6, 15:4, 20:3, 30:2}
max_date_adaptability = dt.datetime(2020,11,22)

#day of birth to compute day of age
birth_date = dt.datetime(2020,6,3)

#nestbox time:
#Nestbox Time 16:00 close / 02:05 open. 
        
#garden opening hours: until (included) date
#The opening of the wintergarden happens automatically but the closing time is written down as soon as the AKB of the first pen is closed. Normally it takes about 10 minutes to close the rest of the AKBs.
date_first_opening_WG = dt.datetime(2020,10,8,0,0,0)
close_dates = [dt.datetime(2021,1,12,0,0,0), dt.datetime(2021,2,1,0,0,0), dt.datetime(2021,2,13,0,0,0), dt.datetime(2021,2,14,0,0,0),
               dt.datetime(2021,2,17,0,0,0)]
dico_garden_opening_hour = {dt.datetime(2020,10,8,0,0,0):{'start_h':11,'start_m':0,'end_h':17,'end_m':0},
                           dt.datetime(2020,10,9,0,0,0):{'start_h':12,'start_m':30,'end_h':17,'end_m':0},
                           dt.datetime(2020,10,11,0,0,0):{'start_h':12,'start_m':30,'end_h':16,'end_m':50},
                           dt.datetime(2020,10,12,0,0,0):{'start_h':13,'start_m':30,'end_h':16,'end_m':50},
                           dt.datetime(2020,10,13,0,0,0):{'start_h':12,'start_m':30,'end_h':17,'end_m':0},
                           dt.datetime(2020,10,14,0,0,0):{'start_h':13,'start_m':30,'end_h':17,'end_m':15},
                           dt.datetime(2020,10,15,0,0,0):{'start_h':12,'start_m':30,'end_h':17,'end_m':0}, #corrected from aviforum
                           dt.datetime(2020,10,16,0,0,0):{'start_h':12,'start_m':30,'end_h':16,'end_m':45},
                           dt.datetime(2020,10,18,0,0,0):{'start_h':12,'start_m':30,'end_h':17,'end_m':15},
                           dt.datetime(2020,10,19,0,0,0):{'start_h':12,'start_m':30,'end_h':17,'end_m':0},
                           dt.datetime(2020,10,20,0,0,0):{'start_h':12,'start_m':30,'end_h':17,'end_m':10},
                           dt.datetime(2020,10,21,0,0,0):{'start_h':11,'start_m':30,'end_h':17,'end_m':10},
                           dt.datetime(2020,10,22,0,0,0):{'start_h':11,'start_m':30,'end_h':17,'end_m':5},
                           dt.datetime(2020,10,23,0,0,0):{'start_h':11,'start_m':30,'end_h':16,'end_m':45}, #corrected from aviforum
                           dt.datetime(2020,10,24,0,0,0):{'start_h':11,'start_m':30,'end_h':16,'end_m':30},
                           dt.datetime(2020,10,25,0,0,0):{'start_h':10,'start_m':30,'end_h':16,'end_m':30},
                           dt.datetime(2020,10,26,0,0,0):{'start_h':10,'start_m':30,'end_h':17,'end_m':0},
                           dt.datetime(2020,10,27,0,0,0):{'start_h':10,'start_m':30,'end_h':16,'end_m':30},
                           dt.datetime(2020,10,28,0,0,0):{'start_h':10,'start_m':30,'end_h':16,'end_m':45},
                           dt.datetime(2020,10,29,0,0,0):{'start_h':10,'start_m':30,'end_h':16,'end_m':0},
                           dt.datetime(2020,10,31,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2020,11,1,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2020,11,4,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2020,11,5,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2020,11,6,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2020,11,7,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2020,11,8,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2020,11,9,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2020,11,10,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':50},
                           dt.datetime(2020,11,11,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2020,11,12,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':35},
                           dt.datetime(2020,11,13,0,0,0):{'start_h':10,'start_m':0,'end_h':11,'end_m':0},
                           dt.datetime(2020,11,14,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2020,11,15,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                           dt.datetime(2020,11,16,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2020,11,17,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2020,11,18,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':35},
                           dt.datetime(2020,11,20,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2020,11,22,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                           dt.datetime(2020,11,23,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':50}, #not real tiem as HA
                           dt.datetime(2020,11,24,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2020,11,25,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2020,11,26,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2020,11,27,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2020,11,28,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2020,11,29,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30}, #was not written by aviforum
                           dt.datetime(2020,11,30,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2020,12,2,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2020,12,3,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':35},
                           dt.datetime(2020,12,4,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2020,12,5,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':50},
                           dt.datetime(2020,12,6,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                           dt.datetime(2020,12,7,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':35},
                           dt.datetime(2020,12,8,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2020,12,9,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2020,12,11,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':35},
                           dt.datetime(2020,12,12,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},#was not written by aviforum
                           dt.datetime(2020,12,13,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2020,12,14,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2020,12,15,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                           dt.datetime(2020,12,16,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2020,12,17,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':50},
                           dt.datetime(2020,12,18,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2020,12,19,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2020,12,20,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2020,12,21,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2020,12,22,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2020,12,23,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':50},
                           dt.datetime(2020,12,27,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2020,12,28,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,1,2,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,1,3,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                           dt.datetime(2021,1,4,0,0,0):{'start_h':10,'start_m':0,'end_h':10,'end_m':1},
                           dt.datetime(2021,1,5,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                           dt.datetime(2021,1,6,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,1,8,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,1,9,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,1,10,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2021,1,11,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},#was not written by aviforum
                           dt.datetime(2021,1,12,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                           dt.datetime(2021,1,13,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,1,14,0,0,0):{'start_h':10,'start_m':10,'end_h':16,'end_m':50},
                           dt.datetime(2021,1,15,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,1,16,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,1,17,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,1,18,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                           dt.datetime(2021,1,19,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,1,20,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':25},
                           dt.datetime(2021,1,21,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,1,22,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,1,23,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2021,1,24,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,1,25,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,1,26,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,1,27,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':35},
                           dt.datetime(2021,1,28,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,1,29,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2021,1,30,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                           dt.datetime(2021,1,31,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                           dt.datetime(2021,2,2,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2021,2,4,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,2,5,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,2,6,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,2,7,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                           dt.datetime(2021,2,8,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,2,9,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,2,10,0,0,0):{'start_h':9,'start_m':10,'end_h':16,'end_m':20},
                           dt.datetime(2021,2,11,0,0,0):{'start_h':11,'start_m':45,'end_h':14,'end_m':0},
                           dt.datetime(2021,2,12,0,0,0):{'start_h':12,'start_m':0,'end_h':13,'end_m':30},
                           dt.datetime(2021,2,15,0,0,0):{'start_h':10,'start_m':40,'end_h':16,'end_m':25},
                           dt.datetime(2021,2,18,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,2,19,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                           dt.datetime(2021,2,20,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2040,2,21,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30}#TODO
                           }
#the keys are included into their values information (i.e. datex:{open time, close time}, date x is opening at that time and closing at that time too (i.e. until (included) date x)

#the date will be removed for all tag with >= lf_counter time having ==0
lf_counter = 5

#went out after its WG_after_opening_mn mn opening of WG?
WG_after_opening_mn = 15

#Number of stayed longer than NonTrans_dur_sec seconds (to remove transitional zones) in each zone 
NonTrans_dur_sec = 60

#min and max date to consider, all other dates will be removed by the "preprocessing_*()" function
date_min = dt.datetime(2020,9,28,0,0,0) 
date_max = min(dt.datetime.now(), dt.datetime(2021,7,25,23,59,59))


dico_HAID_date = {'HA1':dt.datetime(2020,11,23), 
                  'HA2':dt.datetime(2021,1,4), 
                  'HA3':dt.datetime(2021,2,1), 
                  'HA4':dt.datetime(2021,4,12)}

#code associated to each actual readers
dico_rc_sys = {'192.168.1.75':'Reader Pen 3-5',
               #'192.168.1.76':'Reader Pen 6-7',
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
dico_night_hour = {dt.datetime(2020,9,30,0,0,0): {'start_h':9,'start_m':0,'end_h':17,'end_m':0,'nbr_hour':8},
                   dt.datetime(2020,10,8,0,0,0): {'start_h':9,'start_m':0,'end_h':18,'end_m':0,'nbr_hour':9}, #until 8.10.2020 its 9h-18h
                   dt.datetime(2020,10,15,0,0,0): {'start_h':8,'start_m':0,'end_h':18,'end_m':0,'nbr_hour':10},
                   dt.datetime(2020,10,21,0,0,0): {'start_h':7,'start_m':0,'end_h':18,'end_m':0,'nbr_hour':11},
                   dt.datetime(2020,10,23,0,0,0): {'start_h':6,'start_m':0,'end_h':18,'end_m':0,'nbr_hour':12},
                   dt.datetime(2020,10,29,0,0,0): {'start_h':5,'start_m':0,'end_h':17,'end_m':0,'nbr_hour':12},
                   dt.datetime(2020,11,5,0,0,0): {'start_h':4,'start_m':0,'end_h':17,'end_m':0,'nbr_hour':13},
                   dt.datetime(2020,11,12,0,0,0): {'start_h':3,'start_m':0,'end_h':17,'end_m':0,'nbr_hour':14},
                   dt.datetime(2021,8,15,0,0,0): {'start_h':2,'start_m':0,'end_h':17,'end_m':0,'nbr_hour':15}}

#nestbox time that it must stay at least to have a chance biologically to be able to lay an egg in the nextbox
nestbox_sec = 15*60

#TODO: dd nestbox opening! at least for informative purposes

#before this time to compute the number of visists and longest stay (laying egg purpose)
BNestboxHour = 10

#after this time to compute the number of visists and longest stay (hiding purpose)
ANestboxHour = 12

#successful intrusion ratio: (#staid <successfullIntrusionHour h longer than nestbox_sec) / (#of staid <successfullIntrusionHour h)
successfullIntrusionHour = 9

#nbr and % of transition that are starting at zone x going at zone x-j and going again at zone x (j=1,2,3) while staying less than nbr_sec_chaoticmvt_middle seconds in zone x-j
nbr_sec_chaoticmvt_notmiddle = 0 #for now we dont think this should be a constrains: zone x could be for 1secodns or more we dont care
li_nbr_sec_chaoticmvt_middle = [3*60,15*60]

#activity peak: Time of the day when the bird did li_perc_activity% of his total transition of the day (between 0 et 100)
#lower interpolation 
li_perc_activity = [5,25,50,95]

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

#SHOULD NOT BE CHANGED!
#the difference between two timestamp of the time series, and hence will be used for the variables computation and the bining
#for now we set it to one (simpler for bining)
#nbr_sec = 1

#define parameters for variables computations
#restrict time series to one value per ValueDelta seconds
#ValueDelta = 60
#each EntropyTimeComputation values of ts we compute the variables 
#--> compute entropy each EntropyTimeComputation*ValueDelta seconds
#EntropyTimeComputation = 30 
#the shifted variables includes the values of the last NbrData4Shift seconds
#--> each shifted variables will use NbrData/ValueDelta values, while the running will use all previous values starting from 
#the time stamp with at least NbrData/ValueDelta previous values
#NbrData = 60*60*2
#print('we restrict the time series to one value per %d seconds \nwe compute the complexity variables each %d minutes \
#\neach variables includes the values of at least the last %.2f minutes \
#(i.e. are using %.2f values)'%(ValueDelta, (EntropyTimeComputation*ValueDelta)/60, NbrData/60, NbrData/ValueDelta))

#cleaning-model verification: extend the batch until the end baches, to take this into account for duration reliability
dico_BatchID_endhour = {'ID1':16,
                        'ID2':16,
                        'ID3':16,
                        'ID5':13,
                        'ID8':12,
                        'ID9':12,
                        'ID10':16,
                        'ID11':12,
                        'ID12':16,
                        'ID13':13,
                        'ID14':16,
                        'ID17':12,
                        'ID22':10,
                        'ID23':14,
                        'ID25':10,
                        'ID27':14,
                        'ID28':16,
                        'ID30':16,
                        'ID31':16,
                        'ID32':16,
                        'ID37':17}
#the below two dictionaries were made by hand, as at the end I would have verify each output anyway :)
#for the batches that started to be analysed with a record of duration <60sec, we need to know what was the previous zone longer than 1mn
dico_batchID_previoustransitionlonger60sec = {'ID1': '2_Zone',
                                              'ID2': '3_Zone',
                                              'ID11': '4_Zone',
                                              'ID13': '5_Zone',
                                              'ID15': '2_Zone',
                                              'ID16': '3_Zone',
                                              'ID21': '4_Zone',
                                              'ID28': '5_Zone',
                                              'ID32': '5_Zone',
                                              'ID35': '5_Zone',
                                              'ID36': '5_Zone',
                                              'ID40': '2_Zone'}
#find manuallly in the _CLEADEDDATA csv: the model prediction was =1 to the previous zone, which was in 2_Zone
dico_batchID_previoustransitionmodel = {'ID1':'2_Zone'}

#select var
li_cont_select = ['signalstrength', 'signalstzone2','duration_bounded_mn','next_duration_bounded_mn','previous_duration_bounded_mn',
                 'next2zone==Zone','previous2zone==Zone','zone3_match_exist'] 
#duplicate of li_cont_select, all the binary ones
li_bin = ['next2zone==Zone','previous2zone==Zone','zone3_match_exist']
li_cat_select = ['Trackingsystem_Zone','zone2_match','previous1_zone','next1_zone','system']  #PenID
#not pen, as we may not have enough data for this and we dont want to over fit (we want this to be more general


#minimum mvt in 10 minutes to that the 10minutes are categorize as "very low activity during the day"
mvt_counter_min_noactivity = 30 #TODO: if using it, define it better, as the = 90 percentile of all night



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

#zone order&name for visual
#ATTENTION: start from 0 
#ATTENTION: all values of dico_matching should have a key in the dico_zone_order 
#ATTENTION: only the zone from this dictionary will be used!!
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


