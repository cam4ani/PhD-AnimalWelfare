#basic package
import numpy as np
import tqdm
import os
import glob
import sys
import math
import time
import re
import pandas as pd
import random
import json
from collections import defaultdict
import shutil


def extract_info(filename, dico_ID_species, dico_species_ID, path_all_local_image):
    #extract info from title info
    li__ = filename.split('_')
    initial_image_name = '_'.join(li__[2:])
    #remove espace in the name (e.g. why its needed: 10.06.17_151505_ BRE_C, 15.07.07_132316_ABL_A _6)
    initial_image_name = initial_image_name.replace(' ','')
    initial_image_name = initial_image_name.replace('ABl','ABL')
    initial_image_name = initial_image_name.replace('__','_')
    initial_image_name = initial_image_name.replace('..','.')
    initial_image_name = initial_image_name.replace('-','_')
    initial_image_name = initial_image_name.replace('140344BRE','140344_BRE')

    #distinguish all possible case to see any potential error and correct all
    #type 1 : '01.07.15_002046_BRE_B.jpg', '01.07.15_002046_BRE_B_5.jpg', '30.05.17_123559_BAF_E_.png'
    r1 = re.compile('(^\d{2}.\d{2}.\d{2}_\d{6}_[A-Z]{3}_[A-Z].[a-z]{3}$)|(^\d{2}.\d{2}.\d{2}_\d{6}_[A-Z]{3}_[A-Z]_\d+.[a-z]{3}$)|(^\d{2}.\d{2}.\d{2}_\d{6}_[A-Z]{3}_[A-Z]_.[a-z]{3}$)')
    #type 2 : 15.04.15_043257_chevaine.bmp
    r2 = re.compile('^\d{2}.\d{2}.\d{2}_\d{6}_[a-z][a-z]+.[a-z]{3}$')
    
    #,'und' : for undefined
    if (r1.match(initial_image_name) is not None) & (initial_image_name.split('.')[-1] in ['tif', 'jpg', 'png', 'bmp','und']):
        date = initial_image_name[0:8]
        h = int(initial_image_name[9:11])
        m = int(initial_image_name[11:13])
        s = int(initial_image_name[13:15])
        espece_id = initial_image_name[16:19]
        t = initial_image_name[20]
        espece = dico_ID_species[espece_id]
        #check if only one or more fish
        x =  initial_image_name.split('.')[-2]
        if x[-1] in [str(x) for x in range(9)]:
            nbr = int(x.split('_')[-1])
        else:
            nbr = 1

    elif (r2.match(initial_image_name) is not None) & (initial_image_name.split('.')[-1] in ['tif', 'jpg', 'png', 'bmp','und']):
        date = initial_image_name[0:8]
        h = int(initial_image_name[9:11])
        m = int(initial_image_name[11:13])
        s = int(initial_image_name[13:15])
        espece = initial_image_name.split('_')[-1].split('.')[0].upper().replace('TRUITELACUSTRE','TRUITE LACUSTRE').replace('BRÊME','BRÈME')
        espece_id = dico_species_ID[espece]
        espece = dico_ID_species[espece_id]
        t = None
        nbr = np.nan #TODO: ask Damien

    else:
        print(li__[0]+'_'+li__[1]+'_'+initial_image_name)
        return None

    d = {'barrage':li__[0], 'camera':li__[1],
         'local_path':os.path.join(path_all_local_image, filename),
          'local_image_name':filename,
          'initial_image_name':initial_image_name,
          'jour':int(date[0:2]),
          'mois':int(date[3:5]),
          'annee':int('20'+date[6:8]),
          'espece':dico_ID_species[espece_id], #get homogeneous species name (even in type 2)
          'date':date,'heure':h,'minute':m,'seconde':s,'espece_id':espece_id,'taille':t,'nombre':nbr}
    return(d)