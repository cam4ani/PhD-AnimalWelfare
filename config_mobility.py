import os
import numpy as np
import datetime as dt

#besides these configuration file, the csv file must have the following columns: 'Timestamp','Zone','HenID','PenID'

#id of run
#this id should be changed each time you want to generate new results (i.e. be saving everyting without deleting what was already saved. Typically when you modify a function in the "utils.py" file and you would like to compare the results
id_run = 'VF' #'v1'


#choose folder names where the initial data and the extracted data/information should be saved. 
#the extracted_path will be created by the computer if it doesnt exist already
#the initial data path should already exist and have the inital data
path_initial_data = r'D:\vm_exchange\AVIFORUM\data\initial_data_mobility' 
path_extracted_data = r'D:\vm_exchange\AVIFORUM\data\extracted_info_mobility_VF'


li_date2remove = []

#should not change
nbr_sec = 1

#means that all record below and including date 2022.12.15 will have a day from 2:00 to 17:00, and all records from
dico_night_hour = {dt.datetime(3000,1,1,0,0,0):{'start_day_h':2,'start_day_m':0,'end_day_h':17,'end_day_m':0}}
li_hours_in_plot = [2,7,12,17]

#zone order for visual, we should keep interzone as its still some information and also to be sure we always have the same number of hen at each timestamp. All the values should be unique
#ATTENTION: should start from 0 but it can have a hole (e.g. 0,1,3,4 could be the only order of the dico_zone_order)
#ATTENTION: all values of dico_matching (if any) should have a key in the dico_zone_order  
#ATTENTION: only the zone from this dictionary will be used!!
dico_zone_order = {'zone_1':0, 
                   'zone_2':1, 
                   'zone_3':2, 
                   'zone_4':3,
                   'zone_5':4}

dico_zone_plot_name = {'zone_1':'wintergarten', #feed water
                       'zone_2':'litter', #dust
                       'zone_3':'lower tier', #feed water, perches
                       'zone_4':'nestbox', 
                       'zone_5':'upper tier'} #feed water, perches

#should have the same keys as dico_zone_order. can be float
dico_zone_meter2 = {'zone_1':50*50, 
                    'zone_2':3*3, 
                    'zone_3':3*1, 
                    'zone_4':3*0.5,
                    'zone_5':3*0.8}

dico_z_color = {'zone_1':'green', 
                'zone_2':'r', 
                'zone_3':'y', 
                'zone_4':'b',
                'zone_5':'orange',
                np.nan:'white'}

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



#nbr_topics = 6


