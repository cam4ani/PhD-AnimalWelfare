





def aggregate_not_cleaned_df(fct_clean_df, config, max_log_id=None):
    
    '''from a path where the logfiles are (and named 'log_*.csv), it output a df which need to be cleaned, withotu duplicating cleaning if some is already done for several logs and same run_id
    max_log_id is meanly there to test the fct'''
    
    #path where to save extracted data and where to take intiial data
    path_extracted_data = config.path_extracted_data
    path_initial_data = config.path_initial_data
    
    #take all the log files without giving choice so that to avoid error of not startign from initial file
    path_log_files = glob.glob(os.path.join(path_initial_data,'log_*.csv'))
    if max_log_id!=None:
        path_log_files = [i for i in path_log_files if int(i.split('log_')[1].split('.csv')[0])<=int(max_log_id)]
    #id of run (if files were already saved for certain days under that id, then we wont re-clean these days
    id_run = config.id_run
    
    #look which log files were already saved
    p_cleaned = glob.glob(os.path.join(path_extracted_data, id_run+'_clean_records_*.csv'))
    li_saved = [int(j.strip()) for i in p_cleaned for j in i.split('clean_records_')[1].split('.csv')[0].split('_')]
    if len(li_saved)>0:
        print('we already saved log for these days: %s'%' '.join([str(i) for i in sorted(li_saved)]))
    else:
        print('No log files were already cleaned')
    
    print('There is %d log files that was asked to be cleaned'%len(path_log_files))
    #remove all except the last one saved as we will need it to compute correctly the first new logfile
    path_log_files = [i for i in path_log_files if int(i.split('log_')[1].split('.csv')[0]) not in [j for j in li_saved if j!=max(li_saved)]]
    print('There is %d log files that we will use for the cleaning (after removing the one already cleaned except the last one)'%len(path_log_files))
    df = fct_clean_df(path_log_files)
    
    return(df)  


def cleaning_mouvement_records_onlyonce_DEPRECCIATED(config, fct_clean_df, first_last_experiment_logfiles, save=True, max_log_id=None):
    
    '''we will compute everything day by day so taht the experiment can continue indefinitely and we can as time pass already clean the passed days, without cleaning two times the same day. This fct makes the computeation efficient if we want to compute sevearl logfiles in one, but less efficient (but still rapid) when its only one logfile.
    first_last_experiment_logfiles is here to indicate that in that case the logfile will be saved, even if its on the border, otherwise the first and alst wont be saved (indeed, first too)
    NOTE: the interzone will be named accordingly to the first letter of each zone, so you might want to change the zone name before hand (but not changing in a more general wqy, as we still want to be able to verify if each zone is associated to the correct pen'''
    
    #start recording the time it last
    START_TIME = time.clock()
    
    #######################################################################################################################    
    ################# get the data and put into to correct shape    
    #df of all log files to use (i.e. all that exist minus the one already cleaned and uselss to clean the new ones)
    df = aggregate_not_cleaned_df(fct_clean_df, config, max_log_id)
    #create name for saving and list of logfile id to keep for saving and stop the script if none will be saved (i.e. if only 2 borders or all already cleaned)
    df['log_id'] = df['log_id'].astype(int)
    l = df['log_id'].unique()
    #not considering saving the border logids except the border of the whole experiment
    l_not_consider = [i for i in [min(l), max(l)] if i not in [int(k) for k in first_last_experiment_logfiles]]
    print('not considering these logfiles as its at border:', ' '.join([str(i) for i in set(l_not_consider)]))
    li_s = [str(i) for i in df['log_id'].unique() if i not in l_not_consider]
    name_ = '_'.join(li_s)
    if len(li_s)==0:
        print()
        print('No new file to clean'.upper())
        sys.exit()
    print('The name of the new cleaned_log_files is: ',name_)
    
    #######################################################################################################################    
    ################# verify the columns name
    if not all(i in df.columns for i in ['Timestamp','Zone','HenID','PenID','log_id']):
        print('Check your columns name, they should include Timestamp, Zone, HenID, PenID')
        sys.exit()
    print('We Start with %d initial records'%df.shape[0])

    ################# import parameters form configuration file
    id_run = config.id_run
    nbr_sec = config.nbr_sec
    path_extracted_data = config.path_extracted_data
    li_date2remove = config.li_date2remove
    li_henID = config.li_henID
    li_penID = config.li_penID
    dico_zone_matching = config.dico_zone_matching

    #######################################################################################################################
    ################# verify good hen and pen ids
    print('------------------------------------------------------------------------------------------')
    print('Verifying  Hen and Pen ids........')
    print('------------------------------------------------------------------------------------------')

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
                print_color((('BUT not all %s has at least one record (%s)'%(type_,' /-/ '.join([i for i in li if i not in l])),'red'),))
    x0 = df.shape[0]
    df = df[(df['test_correct_HenID']) & (df['test_correct_PenID'])]
    print('We have %d records (%d removed due to wrong hen or pen ids)'%(df.shape[0],x0-df.shape[0]))
    
    ################# add info that is hen making thing more clear and general
    df['HenID'] = df['HenID'].map(lambda x: 'hen_'+x)
    #add day variable
    df['day'] = df['Timestamp'].map(lambda x: dt.datetime(x.year,x.month,x.day))

    #######################################################################################################################
    ################# Remove dates with health care
    print('------------------------------------------------------------------------------------------')
    print('Remove dates with health care........')
    print('------------------------------------------------------------------------------------------')
    li_date2remove = [dt.datetime.strptime(x, '%Y-%m-%d') for x in li_date2remove]
    df['test_toberemoved_date'] = df['day'].map(lambda x: x in li_date2remove)
    x0 = df.shape[0]
    df = df[~df['test_toberemoved_date']]
    print('We have %d records (%d removed due to health-assessment dates)'%(df.shape[0],x0-df.shape[0]))

    ################# sort by date (timestamp), in case the log files were not open in the right order  
    df = df.sort_values(['Timestamp'], ascending=True) #ts_order

    #######################################################################################################################
    ################# remove zone associated to wrong pen
    if dico_zone_matching!=None:
        print('------------------------------------------------------------------------------------------')
        print('remove zone associated to wrong pen........')
        print('------------------------------------------------------------------------------------------')

        df_corr = df.groupby(['PenID','Zone']).count().reset_index()
        df_corr.to_csv(os.path.join(path_extracted_data, id_run+'_Zone_associated_to_pen_record_numbers_'+name_+'.csv'),sep=';')
        #faster than apply : df.apply(lambda x: x['PenID'] in dico_zone_matching[x['Zone']], axis=1)
        df['test_'] = df['PenID']+'/-/'+df['Zone']
        df['test_correct_pen4zone'] = df['test_'].map(lambda x: x.split('/-/')[0] in dico_zone_matching[x.split('/-/')[1]])
        df_corr = df[~df['test_correct_pen4zone']]
        df_corr.to_csv(os.path.join(path_extracted_data,id_run+'_Zone_associated_to_wrong_Pen_all_situation_'+name_+'.csv'),sep=';')
        x0 = df.shape[0]
        df = df[df['test_correct_pen4zone']]
        print('We have %d records (%d removed due zone associated to wrong pen)'%(df.shape[0],x0-df.shape[0]))
    
    #######################################################################################################################
    ################# handle flickering situations
    #A flickering situation happens when a hen change zone within strictly less than 2seconds, in which case we name these 
    #situations "Interzone" and keep only the first timestamp of each interzones situation
    print('------------------------------------------------------------------------------------------')
    print('Handle flickering situations........')
    print('------------------------------------------------------------------------------------------')

    #add next duration variable
    li_df = []
    #more efficient to do it per hen, as it wont need to search in the whole dataframe, and we can simply shift the timestamp 
    #column
    print('First we will compute the previous duration and flag the flickering situations as interzones')
    for i, df_hen in tqdm.tqdm(df.groupby(['HenID'])):

        #as the next record date (sort by date, then simply shift by one row and add nan at then end)
        df_hen = df_hen.sort_values(['Timestamp'], ascending=True) #ts_order

        #same date, one must take the last recorded one & sorting by date might change it. Also it already should be sorted by 
        #date
        df_hen['next_record_date'] = df_hen['Timestamp'].tolist()[1:]+[np.nan]
        #compute duration
        df_hen['duration'] = df_hen.apply(lambda x: x['next_record_date']-x['Timestamp'], axis=1)

        #compute the last record date in order to put interzone also when the duration is >=nbr_sec
        df_hen['previous_record_date'] = [np.nan]+df_hen['Timestamp'].tolist()[0:-1]
        #compute previous duration in order to put interzone also when the duration is >=nbr_sec
        df_hen['previous_duration'] = [np.nan]+df_hen['duration'].tolist()[0:-1]

        li_df.append(df_hen)
    #put again in one dataframe
    df = pd.concat(li_df)
    #dont care about the false positive warning 
    
    #name it interzone when duration is less than 2 seconds
    #note that there is no need to merge interzone in one timestamp as we will in any case extend to a time serie for analysis and
    #keep only one second over two
    df['Zone_without_flickering'] = df['Zone'].copy()
    df.loc[df['duration']<dt.timedelta(seconds=nbr_sec),'Zone_without_flickering'] = 'Interzone'
    #if its not interzone (i.e. its duration is longer than 3 seconds) and its previous duration is shorter than 3 seconds,
    #then its the end of a flickering situation
    df.loc[(df['previous_duration']<dt.timedelta(seconds=nbr_sec))&(df['Zone_without_flickering']!='Interzone'),
           'Zone_without_flickering'] = 'Interzone_f'
                                                                            
    #we wont be doing this, as otherwise if the lasst timestamp are flickering situation, we will miss a zone
        #replace 'Zone_without_flickering' by np.nan if the duration is nan (i.e. if last observation)
    #df.loc[pd.isnull(df['duration']),'Zone_without_flickering'] = np.nan

    print('Secondly we will distinguish between different interzone, given different names which is \
    very long as we need to look at the next zone only once we have changed the next row') 
    #differentiate between different interzones
    #very long as we need to look at the next zone only once we have changed the next row
    li_df = []
    for i, df_hen in tqdm.tqdm(df.groupby(['HenID'])):

        df_hen['interzone_info'] = df_hen['Zone'].copy()
        df_hen['test_ToRemove'] = False
        df_hen = df_hen.reset_index()
        #as we will keep the first entry of consecutives equal zones, we will take the value from the next record to the previous 
        #one (and not the opposite way)
        #idea: keep the first interzone entry, and put all info about these interzones in the interzone_info column
        #start with the last row, and put the rule: if row is not the same as the one after then just add the actual zone as info,
        #otherwise if its the same then add the interzone info of now and after
        for i in reversed(range(1,df_hen.shape[0]-1)):
            xp = df_hen.iloc[i-1]
            x0 = df_hen.iloc[i]
            x1 = df_hen.iloc[i+1] #--> we can not deal with the last row of each df_hen, so we will remove it
            if (x0['Zone_without_flickering']!=x1['Zone_without_flickering']) & (x1['Zone_without_flickering']!='Interzone_f'):
                df_hen.at[i,'interzone_info'] = x0['Zone'] 
            else:
                df_hen.at[i,'interzone_info'] = x0['interzone_info']+', '+x1['interzone_info']

            #if its interzone and the previous one is interzone then remove it
            if ((x0['Zone_without_flickering']=='Interzone')|(x0['Zone_without_flickering']=='Interzone_f'))\
            & (xp['Zone_without_flickering']=='Interzone'):
                df_hen.at[i,'test_ToRemove'] = True
        #remove last row 
        df_hen = df_hen[0:-1]
        li_df.append(df_hen)
    #put again in one dataframe
    df = pd.concat(li_df)
    #dont care about the false positive warning

    #correct names and keep the first interzone row
    df['Zone_without_flickering_nodiff'] = df['Zone_without_flickering'].copy()
    #df['interzone_name'] = df['interzone_info'].map(lambda x: 'Interzone_'+\
    #                                                ''.join(sorted([j[0] for j in set([i.strip() for i in x.split(',')])])) )
    df['interzone_name'] = df['interzone_info'].map(lambda x: 'Interzone_'+\
                                                    ''.join(sorted([str(dico_matching[j][0]) for j in set([i.strip() for i in x.split(',')])])) )    
    df['Zone_without_flickering'] = np.where(df['Zone_without_flickering']=='Interzone', 
                                             df['interzone_name'], 
                                             df['Zone_without_flickering'])

    df.to_csv(os.path.join(path_extracted_data,id_run+'_records_before_removing_flickering_situtation_'+name_+'.csv'), index=False,sep=';')

    #we remove NOW the interzones record that are not the first record and NOT BEFOREHAND, in order to verify things alltogether
    x0 = df.shape[0]
    df = df[df['test_ToRemove']==False]
    #Now, duration/previous_record_date/next_record_date makes no more sense for interzone and consecutives equal records, 
    #so lets remove it
    df.drop(['duration','previous_duration','previous_record_date','Zone_without_flickering_nodiff',
            'interzone_name','index'], inplace=True, axis=1)
    print('We have %d records (%d removed due to flickering situations)'%(df.shape[0],x0-df.shape[0]))  
    
    #######################################################################################################################
    ################# remove consecutives equal Zone for same hens at (not strictly) more than nbr_second second duration
    #add next zone based on Zone_without_flickering (for quality verification)
    print('------------------------------------------------------------------------------------------')
    print('remove consecutives equal Zone for same hens at (not strictly) more than nbr_second second duration........')
    print('------------------------------------------------------------------------------------------')

    li_df = []
    #more efficient to do it per hen, as it wont need to search in the whole dataframe, and we can simply shift the timestamp column
    for i, df_hen in tqdm.tqdm(df.groupby(['HenID'])):

        #as the next record date (sort by date, then simply shift by one row and add nan at then end)
        df_hen = df_hen.sort_values(['Timestamp'], ascending=True) #ts_order
        #same date, one must take the last recorded one & sorting by date might change it. Also it already shoul dbe sorted by date
        df_hen['next_zone'] = df_hen['Zone_without_flickering'].tolist()[1:]+[np.nan]

        li_df.append(df_hen)

    #put again in one dataframe
    df = pd.concat(li_df)
    #dont care about the false positive warning

    #True if next zone is equal to the actual zone (Zone_without_flickering) 
    df['correction_is_consecutive_equal_zone'] = False
    df['Zone_without_flickering'] = df['Zone_without_flickering'].fillna('')
    df['is_flick'] = df['Zone_without_flickering'].map(lambda x: x.startswith('Interzone') if x!='' else False)
    df.loc[(df['next_zone']==df['Zone_without_flickering']) & (~df['is_flick']), 'correction_is_consecutive_equal_zone'] = True
    df_test = df[df['correction_is_consecutive_equal_zone']]
    print_color((('There is ','black'), (df_test.shape[0],'red'), ('records that has same zone than the previous one without being a flickering situation, we wont remove these records from the clean_record file as it wont change anything for the time serie','black')))
    df_test.filter(['HenID','Timestamp','Zone_without_flickering','next_record_date',
                    'next_zone']).to_csv(os.path.join(path_extracted_data,id_run+'_consecutives_equal_zone_'+name_+'.csv'), 
                                         index=False, sep=';')
    print('We have %d records'%df.shape[0])

    #as we simply want to keep the first one, its the same as not doing anything. In case we would do something, then we would
    #also differencitae the interzone name here
    #removing unnecessary columns
    df.drop(['next_record_date','next_zone'], inplace=True, axis=1)

    #######################################################################################################################
    ################# last cleaning and saving
    #removing unnecessary columns
    df.drop([c for c in df.columns if c.startswith('test_')], inplace=True, axis=1)
    
    #we will save all except the first and last logfile, except if its the first or last one of the whole experiment
    df = df[~df['log_id'].isin(l_not_consider)]
    
    #save
    if save:
        df.to_csv(os.path.join(path_extracted_data, id_run+'_CleanRecords'+name_+'.csv'), sep=';', index=False)
        
    END_TIME = time.clock()
    print ("Total running time: %.2f mn" %((END_TIME-START_TIME)/60))
    return(df)
    
    
