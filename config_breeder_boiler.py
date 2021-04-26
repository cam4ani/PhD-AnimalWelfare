import os
import numpy as np
import datetime as dt


#id of run
#this id should be changed each time you want to generate new results (i.e. be saving everyting without deleting what was already saved. Typically when you modify a function in the "utils.py" file and you would like to compare the results
id_run = 'FINAL_' #'v1'


#choose folder names where the initial data and the extracted data/information should be saved. 
#the extracted_path will be created by the computer if it doesnt exist already
#the initial data path should already exist and have the data in the following form:
#path_extracted_data\log_*
path_initial_data = r'D:\vm_exchange\AVIFORUM\data\initial_data_bb'
path_extracted_data = r'D:\vm_exchange\AVIFORUM\data\extracted_info_breeder_broiler_FINAL'


#list of date to be removed due to health assessment or anything else
li_date2remove = [dt.datetime(2019,7,9), dt.datetime(2019,7,10), dt.datetime(2019,7,16), dt.datetime(2019,7,17),
                  dt.datetime(2019,7,23), dt.datetime(2019,7,24), dt.datetime(2019,7,30), dt.datetime(2019,7,31),
                  dt.datetime(2019,8,6), dt.datetime(2019,8,7), dt.datetime(2019,8,13), dt.datetime(2019,8,14),
                  dt.datetime(2019,8,19), dt.datetime(2019,8,20), dt.datetime(2019,8,21)]


#list of HenID and PenID that should exists, everything not in this list will not be taken into account and a wargnin will be send with
#all the associated information. To stay general, new rules about henid and penid correction should be addedd before using the functions
#should be string, but will be converted into string in anycase
li_henID = ['4B', '4L', '1A', '2L', '4H', '4M','1L', '6B', '8A', '6H', '7S', '8X',
            '6F', '10F', '1X', '6V', '3S', '7H','8B', '2M', '1H', '3C', '4A', '9F',
            '4V', '2F', '4X', '3M', '8S', '2B','2X', '1C', '9S', '2V', '6A', '3X',
            '3A', '2A', '2S', '6L', '4S', '6X','10V', '10X', '8M', '5M', '6C', '6S',
            '6M', '8V', '1B', '5V', '8L', '8H','9L', '4F', '8C', '2H', '9A', '3F',
            '10M', '1S', '7F', '2C', '9H', '8F','1V', '10H', '10L', '5X', '10C', '10B',
            '5L', '10A', '7L', '7A', '1M', '9V','5F', '7B', '9X', '7V', '5A', '7X',
            '7M', '9C', '3H', '9M', '3V', '5H','9B', '5S', '3B', '4C', '5C', '5B',
            '1F', '10S', '7C', '3L']
li_penID = ['1','2','3','4','5','6','7','8','9','10']


#list of flickering type 2 nbr_block value (3 means at least 3 time brbrbr)
#li_nbr_block_repetition = [3,5,7]

#list of hours that must be associated to night, and to day, example:
#dico_night_hour = {dt.datetime(2019,11,15,0,0,0):{'start_day_h':3,'start_day_m':5,'end_day_h':17,'end_day_m':25},
#                   dt.datetime(2019,12,15,0,0,0):{'start_day_h':3,'start_day_m':0,'end_day_h':16,'end_day_m':15}}
#means that all record below and including date 2019.11.15 will have a day from 3:05 to 17:25, and all records from
#]2019-11-15 to 2019-12-15] will have a day from 3:00 to 16:15
#dico_night_hour = {dt.datetime(2022,12,15,0,0,0):{'start_day_h':3,'start_day_m':0,'end_day_h':16,'end_day_m':0},}
#li_day_hours = [3,4,5,6,7,8,9,10,11,12,13,14,15]

#min and max date to consider, other date will be removed by the "preprocessing_*()" function
date_min = None #dt.datetime(2019,7,11,0,0,0)
date_max = None #dt.datetime(2019,7,14,0,0,0)
#date_min = dt.datetime(2018,7,11,0,0,0)
#date_max = dt.datetime(2021,7,14,0,0,0)


#maximum number of different zone to be included in a flickering type2 event
nbrZone = 2
#
#number of seconds that a hens needs to stay at least in a zone to count as a true transition for flickering type 1
#note that if nbr_sec =3, then from 2019-07-07 22:09:32 to 2019-07-07 22:09:35 , there is at least 2 seconds so its consider
#as a transition, but not 32 - 34 
#nbr_sec_flickering1 = 3

#the difference between two timestamp of the time series, and hence will be used for the variables computation and the bining
#for now we set it to one (simpler for bining)
#nbr_sec = 1

#pen associated to zone
dico_zone_matching = {'Rampe 1-5': ['1','2','3','4','5'], 
                      'Einstreu 1-5':['1','2','3','4','5'], 
                      'Rampe 6-10':['6','7','8','9','10'],
                      'Einstreu 6-10':['6','7','8','9','10'], 
                      'Box1':['1'],  
                      'Box2':['1'], 
                      'Box3':['2'], 
                      'Box4':['2'], 
                      'Box5':['3'], 
                      'Box6':['3'], 
                      'Box7':['4'], 
                      'Box8':['4'], 
                      'Box9':['5'], 
                      'Box10':['5'], 
                      'Box11':['6'], 
                      'Box12':['6'], 
                      'Box13':['7'], 
                      'Box14':['7'], 
                      'Box15':['8'], 
                      'Box16':['8'], 
                      'Box17':['9'], 
                      'Box18':['9'], 
                      'Box19':['10'], 
                      'Box20':['10'], 
                      np.nan:['1','2','3','4','5','6','7','8','9','10']}


#initial zone matching to general names, will be used in the "general_cleaning()" function
#ATTENTION: first character of the new name should be unique (will be used in flickering situation, to find pattern (e.f. brbrbr) etc)
dico_matching = {'Box1': 'A Box',
                 'Box10': 'B Box',
                 'Box11': 'A Box',
                 'Box12': 'B Box',
                 'Box13': 'A Box',
                 'Box14': 'B Box',
                 'Box15': 'A Box',
                 'Box16': 'B Box',
                 'Box17': 'A Box',
                 'Box18': 'B Box',
                 'Box19': 'A Box',
                 'Box2': 'B Box',
                 'Box20': 'B Box',
                 'Box3': 'A Box',
                 'Box4': 'B Box',
                 'Box5': 'A Box',
                 'Box6': 'B Box',
                 'Box7': 'A Box',
                 'Box8': 'B Box',
                 'Box9': 'A Box',
                 'Einstreu 1-5': 'Einstreu',
                 'Einstreu 6-10': 'Einstreu',
                 'Rampe 1-5': 'Rampe',
                 'Rampe 6-10': 'Rampe'}

#should be one of the new name from dico_matching. Used in "verification_based_on_initial_record()" function. None in the breeder boiler
#as there is no outside zone
#outside_zone = None

#define the impossible mouvement from one zone to another, consider only the first caractere of the zones defined by dico_matching
#each possible value should be put as a value, with empty list if no impossible movement is possible
#dico_impossible_mvt_to = {'A':['B','E'],
#                          'B':['A','E'],
#                          'E':['A','B'],
#                          'R':[]}

#set of zone that shoul dnot be identified by flickering 2 (i.e. when they would be, and if thats actually correct sequence, then they
#might be very long sequences like er, which will slow down the process
#if computing the flickering type2 takes too long then you should probably think if you did not miss some possible combination
#leave an empty list if no such case apply to your data
#li_not_flickering2 = [set(['E','R']),]

#zone order for visual, we should keep interzone as its stills ome information and also to be sure we always have the same number of hen at each timestamp. All the values should be unique
#ATTENTION: start from 0 
#ATTENTION: all values of dico_matching should have a key in the dico_zone_order  
#ATTENTION: only the zone from this dictionary will be used!!
#dico_zone_order = {'A Box':0,
#                   'B Box':1,
#                   'Rampe':2,
#                   'Einstreu':3}


