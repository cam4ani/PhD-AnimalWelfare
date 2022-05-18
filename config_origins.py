import os
import numpy as np
import datetime as dt
from collections import defaultdict
import pandas as pd

#here d'ete 28.03.2021!!
#ATTENTION: Taking out the manure/shit this week will not happen on Thursday 24.12 but on Wednesday (23.12.)

#email: 07/07/2021, 12:20the barn is always treated against mites before the birds move in (as a prevention). A second round of treating #against mites is normally only performed when needed. 

#id of run
#this id should be changed each time you want to generate new results (i.e. be saving everyting without deleting what was already saved. Typically when you modify a function in the "utils.py" file and you would like to compare the results
id_run = 'ALLDATA_'
#id_run = 'correctlightschedule_'
#other available: chapter0_final_

#choose folder names where the initial data and the extracted data/information should be saved. 
#the extracted_path will be created by the computer if it doesnt exist already
#the initial data path should already exist and have the data in the following form:
#path_initial_data\Abteile 3_5\log_*
#path_initial_data\Abteile 10_12\log_*
#CSV
#focal_name = 'FocalBirdsInfo26-07-2021.csv'
#day_name = 'TrackingDaysWhenCanWeUseWhat.csv'
focal_name = 'FocalBirdsInfo26-07-2021_comma.csv'
day_name = 'TrackingDaysWhenCanWeUseWhat_comma.csv'
path_initial_data = r'G:\VPHI\Welfare\2- Research Projects\OFHE2.OriginsE2\GantnerSystem\_dailycheckingSystem'
path_dataoutput = r'G:\VPHI\Welfare\2- Research Projects\OFHE2.OriginsE2\DataOutput'
#path_initial_data = r'R:\OriginsProject\OFHE2.OriginsE2\GantnerSystem\_dailycheckingSystem'
#path_dataoutput = r'R:\OriginsProject\OFHE2.OriginsE2\DataOutput'
path_extracted_data = os.path.join(path_dataoutput,'TrackingSystem') 
path_extracted_HA = os.path.join(path_dataoutput,'HA') 

path_FocalBird = os.path.join( r'G:\VPHI\Welfare\2- Research Projects\OFHE2.OriginsE2','FOCAL BIRDS',focal_name)
path_Days = os.path.join( r'G:\VPHI\Welfare\2- Research Projects\OFHE2.OriginsE2','FOCAL BIRDS',day_name)
path_performance = os.path.join(r'G:\VPHI\Welfare\2- Research Projects\OFHE2.OriginsE2\Productivity')
#path_FocalBird = os.path.join( r'R:\OriginsProject\OFHE2.OriginsE2','FOCAL BIRDS',focal_name)
#path_Days = os.path.join( r'R:\OriginsProject\OFHE2.OriginsE2','FOCAL BIRDS',day_name)
#path_performance = os.path.join(r'R:\OriginsProject\OFHE2.OriginsE2\Productivity')

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
               'pen6':'TRAN-nonfocal', 'pen7':'OFH-nonfocal',
              3:'OFH',5:'OFH',9:'OFH',11:'OFH',
               4:'TRAN',8:'TRAN',10:'TRAN',12:'TRAN',
               6:'TRAN-nonfocal', 7:'OFH-nonfocal'}
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
date_populationday = dt.datetime(2020,9,29)
max_date_adaptability = dt.datetime(2020,11,22)
min_date_drivers = dt.datetime(2020,11,14) #light schedule finished to change

#day of birth to compute day of age
birth_date = dt.datetime(2020,6,3) #DOA 1 = 2020-6-4

#from Markus email on the 9 march 2022:
#Nestbox Time 16:00 close / 02:05 open. 
#eggs collections:Normally we start at 07:45 – 08:00 Uhr collecting eggs on the floor, but can be different at the weekends. After this, 
#we start the egg collection in the front room

#vaccination date
#typically on a vaccination day the water will be turned off from around 8h to 10h30, then the vaccination will be delivered for like two hours through the water. It shouldnt taste anything
dico_vaccinationDate_type = {dt.datetime(2020,11,13):'Ecoli', #wg finish earlier, and first date barn schedule regular so we wont have it, only on flock that needs it, and our flock needed it
                             dt.datetime(2020,12,30):'IBMa5Nobilis', #wg 13h, we have mvt data
                             dt.datetime(2021,1,12):'IB4/91Nobilis', #wg close but not certain from aviforum anotation, no mvt data 
                             dt.datetime(2021,3,9):'IBMa5Nobilis', #wg 13h, we have mvt data
                             dt.datetime(2021,3,26):'IB4/91Nobilis', #wg 13h, we have mvt data
                             dt.datetime(2021,5,7):'IBMa5Nobilis',#wg 13h, we have mvt data
                             dt.datetime(2021,5,21):'IB4/91Nobilis'} #wg close

#first day when light started at 2h in the morning
date_consistent_barn_schedule = dt.datetime(2020,11,13)

#dawn and dusk light schedule only form chapter two: i.e. only since barn schedule is stable (i.e. the above date)
li_light_dawn_ = pd.date_range(start=dt.datetime(2020,1,1,2,0,0), end=dt.datetime(2020,1,1,2,4,0), freq = 'S')
li_light_dawn = pd.date_range(start=dt.datetime(2020,1,1,2,0,0), end=dt.datetime(2020,1,1,16,59,59), freq = 'S')
li_light_dawn = [1 if x in li_light_dawn_ else 0 for x in li_light_dawn] 
li_light_dusk_ = pd.date_range(start=dt.datetime(2020,1,1,16,47,0), end=dt.datetime(2020,1,1,17,0,0), freq = 'S')
li_light_dusk = pd.date_range(start=dt.datetime(2020,1,1,2,0,0), end=dt.datetime(2020,1,1,16,59,59), freq = 'S')
li_light_dusk = [1 if x in li_light_dusk_ else 0 for x in li_light_dusk] 

########################################################################################
#min_h and max_h of laying egg behavior and of hiding behavior
tuple_min_max_egglaying_h = (2,6)
tuple_min_max_egghiding_h = (10,15) #all hours after x:0:0 and until y:59:50

#feeding running at x:y for dur_FR_beforeandafter_mn before and after that time. For power reason, its per group of four pens, however, the noise as stimuli may be in pen 5 when the pen1 - pen4 are running, though the other pens will be running 1mn later. it starts with pen -4, then pen 5-8, etc : 5 groups of 4 pens, each running for 1 mn
#note that there is also a sensor in the feeding system that sense if there is enough food in the big container, and if not it will fill this up. this can happen anytime, and will make some noise, but unfortunately we cannot have this information
#NOTE: this is only since date_consistent_barn_schedule! else its varying
tupleFR_h_mn = [(2,31),(6,1),(9,1),(12,1),(14,16),(16,16)] 
#duration of feeding line before and after the time given in tupleFR_h_mn
dur_FR_beforeandafter_mn = 2
#duration before and after food runing that is defined as grey area: with mixed behavior of waiting for food vs not waiting for food
dur_around_FR_2remove = 15

#li_FR: 1 when food is running (else 0) ; li_FNR: 1 when food is not running (else 0)
#compute list of 0 (food not running)/1(=food runing), with one value per second
li_when_food_running = []
for h,mn in tupleFR_h_mn:
    li_when_food_running.extend(pd.date_range(start=dt.datetime(2020,1,1,h,mn,0)-dt.timedelta(minutes=dur_FR_beforeandafter_mn), 
                                      end=dt.datetime(2020,1,1,h,mn,0)+dt.timedelta(minutes=dur_FR_beforeandafter_mn), 
                                      freq = 'S'))
li_FR = pd.date_range(start=dt.datetime(2020,1,1,2,0,0), end=dt.datetime(2020,1,1,16,59,59), freq = 'S') 
li_FR = [1 if x in li_when_food_running else 0 for x in li_FR]
li_when_food_notnotrunning = [] ## all except when running or grey zone
for h,mn in tupleFR_h_mn:
    li_when_food_notnotrunning.extend(pd.date_range(start=dt.datetime(2020,1,1,h,mn,
                                                                0)-dt.timedelta(minutes=dur_FR_beforeandafter_mn+dur_around_FR_2remove), 
                                      end=dt.datetime(2020,1,1,h,mn,0)+dt.timedelta(minutes=dur_FR_beforeandafter_mn+dur_around_FR_2remove), 
                                      freq = 'S'))
li_FNR = pd.date_range(start=dt.datetime(2020,1,1,2,0,0), end=dt.datetime(2020,1,1,16,59,59), freq = 'S')
li_FNR = [0 if x in li_when_food_notnotrunning else 1 for x in li_FNR] 
#small visual verification
#plt.plot(li_FR);

#compute list of 0 (not laying behavior)/1(=laying behavior), with one value per secon
li_timeforlaying = pd.date_range(start=dt.datetime(2020,1,1,tuple_min_max_egglaying_h[0],0,0), 
                                      end=dt.datetime(2020,1,1,tuple_min_max_egglaying_h[1],0,0), 
                                      freq = 'S')
li_LT = pd.date_range(start=dt.datetime(2020,1,1,2,0,0), end=dt.datetime(2020,1,1,16,59,59), freq = 'S') 
li_LT = [1 if x in li_timeforlaying else 0 for x in li_LT]


#compute list of 0 (not hiding behavior)/1(=hiding behavior), with one value per secon
li_timeforhiding = pd.date_range(start=dt.datetime(2020,1,1,tuple_min_max_egghiding_h[0],0,0), 
                                      end=dt.datetime(2020,1,1,tuple_min_max_egghiding_h[1],0,0), 
                                      freq = 'S')
li_HT = pd.date_range(start=dt.datetime(2020,1,1,2,0,0), end=dt.datetime(2020,1,1,16,59,59), freq = 'S') 
li_HT = [1 if x in li_timeforhiding else 0 for x in li_HT]    
########################################################################################


#garden opening hours: until (included) date
#The opening of the wintergarden happens automatically but the closing time is written down as soon as the AKB of the first pen is closed. Normally it takes about 10 minutes to close the rest of the AKBs.
date_first_opening_WG = dt.datetime(2020,10,8,0,0,0)
#dates that the wg was close the entire day
close_dates = [dt.datetime(2021,1,12,0,0,0), dt.datetime(2021,2,1,0,0,0), dt.datetime(2021,2,13,0,0,0), dt.datetime(2021,2,14,0,0,0),
               dt.datetime(2021,2,17,0,0,0),dt.datetime(2021,5,21,0,0,0)]
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
                           dt.datetime(2020,11,23,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':50}, #not real time as HA
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
                           dt.datetime(2020,12,29,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2020,12,30,0,0,0):{'start_h':13,'start_m':0,'end_h':16,'end_m':30},#vaccination
                           dt.datetime(2020,12,31,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
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
                           dt.datetime(2021,2,11,0,0,0):{'start_h':11,'start_m':45,'end_h':14,'end_m':0}, #because very cold
                           dt.datetime(2021,2,12,0,0,0):{'start_h':12,'start_m':0,'end_h':13,'end_m':30}, #because very cold
                           dt.datetime(2021,2,15,0,0,0):{'start_h':10,'start_m':40,'end_h':16,'end_m':25},
                           dt.datetime(2021,2,18,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,2,19,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                           dt.datetime(2021,2,20,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15}, #TODO
                           dt.datetime(2021,2,21,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,2,22,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,2,23,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                           dt.datetime(2021,2,24,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                           dt.datetime(2021,2,25,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2021,2,26,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                           dt.datetime(2021,2,27,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,2,28,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2021,3,1,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':50},
                           dt.datetime(2021,3,2,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':35},
                           dt.datetime(2021,3,3,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2021,3,4,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':50},
                           dt.datetime(2021,3,5,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,3,6,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':50},
                           dt.datetime(2021,3,7,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2021,3,8,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,3,9,0,0,0):{'start_h':13,'start_m':30,'end_h':16,'end_m':30},#vaccination
                           dt.datetime(2021,3,10,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,3,11,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,3,12,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,3,13,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2021,3,14,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2021,3,15,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,3,16,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,3,17,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2021,3,18,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,3,19,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':35},
                           dt.datetime(2021,3,20,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2021,3,21,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                           dt.datetime(2021,3,22,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2021,3,23,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,3,24,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                           dt.datetime(2021,3,25,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,3,26,0,0,0):{'start_h':13,'start_m':30,'end_h':16,'end_m':30},#vaccination
                           dt.datetime(2021,3,27,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,3,28,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,3,29,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,3,30,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,3,31,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':25},
                           dt.datetime(2021,4,1,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2021,4,2,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,4,3,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,4,4,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,4,5,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,4,6,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,4,7,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,4,8,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,4,9,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,4,10,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,4,11,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,4,12,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,4,13,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,4,14,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,4,15,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':25},
                           dt.datetime(2021,4,16,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                           dt.datetime(2021,4,17,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                           dt.datetime(2021,4,18,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':5},
                           dt.datetime(2021,4,19,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,4,20,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,4,21,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                           dt.datetime(2021,4,22,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,4,23,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,4,24,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,4,25,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2021,4,26,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,4,27,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,4,28,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,4,29,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,4,30,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                           dt.datetime(2021,5,1,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,5,2,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,5,3,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':35},
                           dt.datetime(2021,5,4,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2021,5,5,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2021,5,6,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,5,7,0,0,0):{'start_h':13,'start_m':30,'end_h':16,'end_m':30},#vaccination
                           dt.datetime(2021,5,8,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,5,9,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,5,10,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,5,11,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':25},
                           dt.datetime(2021,5,12,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,5,13,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,5,14,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,5,15,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,5,16,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,5,17,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,5,18,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':25},
                           dt.datetime(2021,5,19,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':25},
                           dt.datetime(2021,5,20,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':25},
                           dt.datetime(2021,5,21,0,0,0):{'start_h':1,'start_m':0,'end_h':1,'end_m':1}, #NO WG
                           dt.datetime(2021,5,22,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,5,23,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2021,5,24,0,0,0):{'start_h':10,'start_m':0,'end_h':15,'end_m':30},
                           dt.datetime(2021,5,25,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':45},
                           dt.datetime(2021,5,26,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,5,27,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,5,28,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,5,29,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,5,30,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,5,31,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},         
                           dt.datetime(2021,6,1,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,6,2,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,6,3,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,6,4,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,6,5,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2021,6,6,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,6,7,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2021,6,8,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':35},
                           dt.datetime(2021,6,9,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':25},
                           dt.datetime(2021,6,10,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,6,11,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,6,12,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                           dt.datetime(2021,6,13,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,6,14,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,6,15,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,6,16,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,6,17,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2021,6,18,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2021,6,19,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,6,20,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':15},
                           dt.datetime(2021,6,21,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2021,6,22,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,6,23,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,6,24,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':35},
                           dt.datetime(2021,6,25,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,6,26,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,6,27,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                           dt.datetime(2021,6,28,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                           dt.datetime(2021,6,29,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,6,30,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,7,1,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2021,7,8,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,7,9,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':25},
                           dt.datetime(2021,7,10,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':0},
                           dt.datetime(2021,7,11,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':10},
                           dt.datetime(2021,7,12,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,7,13,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,7,14,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,7,15,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,7,16,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':35},
                           dt.datetime(2021,7,17,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':50},
                           dt.datetime(2021,7,18,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':30}, #not written in aviforum planning
                           dt.datetime(2021,7,19,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':46},
                           dt.datetime(2021,7,20,0,0,0):{'start_h':10,'start_m':0,'end_h':16,'end_m':40},
                           dt.datetime(2021,7,21,0,0,0):{'start_h':8,'start_m':0,'end_h':16,'end_m':30},
                           dt.datetime(2021,7,22,0,0,0):{'start_h':8,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,7,23,0,0,0):{'start_h':8,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,7,24,0,0,0):{'start_h':8,'start_m':0,'end_h':16,'end_m':20},
                           dt.datetime(2021,7,25,0,0,0):{'start_h':8,'start_m':0,'end_h':17,'end_m':0},
                           dt.datetime(2021,7,26,0,0,0):{'start_h':8,'start_m':0,'end_h':16,'end_m':30}                          }
#the keys are included into their values information (i.e. datex:{open time, close time}, date x is opening at that time and closing at that time too (i.e. until (included) date x)


#min duration in seconds to account for in the count of visitis to Z4 in the morning
mindur_toaccountforZ4 = 15


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
                  'HA4':dt.datetime(2021,4,12), 
                  'HA5':dt.datetime(2021,7,25)} #26 wont recognize the bird as tracking stop on the 25. we did HA on 26 but last day of tracking is 25

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

#from chapter 2: chickens get stable from datetime.datetime(2020, 11, 8, 0, 0): 40 days in the barn
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
li_nbr_sec_chaoticmvt_middle = [3*60] #15*60

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


