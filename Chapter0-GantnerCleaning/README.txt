------------------------------------------ Datasets
------------------- 1
"Sample-Traking-Data.csv": Sample of the data

------------------- 2
"Video_Observation_training&testing.csv": Video observations results for both training and tst dataset,with the following annotation code:
    -1: can't be sure (e.g. not clear on camera)
    1: CR - correct record cf terminology described in paper
    0: WR - wrong record cf terminology described in paper
    3: an actual transition that was missed by our tracking system
    Any term describing a zone name (e.g. Tier 1, Tier2, winter garden, ...): an actual transition that was missed by our tracking system

Out of the 48 batches initially chosen for the verification dataset, we ended up having 42 batches due to 
•cameras orientation issues, 
•backpack not being clearly distinguishable and 
•birds that did not moved during the entire batch and for which we had no evidence if the tag functioned correctly (therefore were removed from the batches)

Training dataset further details: a total of 46 tags is involved in the training dataset, with 17 tags involved with more than 100 observations and 35 tags involved in more than 40 observations. The training dataset contains 1’740 observations from the tracking system involving pen 3-5, 1’407 observations from the tracking system involving pens 10-12, and 1’127 observations from the tracking system involving pens 8-9. The training dataset contains 916 observations from the litter area, 2’393 observations from the lower perch, 404 observations from the nestbox zone and 561 observations from the top floor.


------------------------------------------ Decision tree classifiers
1_PreprocessedTraining&TestingData.ipynb: merge the records metadata with the video analysis. output: "id_run+InputCleaning.csv"

2_Classifier-RF&GradientBoosting_finetuning.ipynb: use "id_run+InputCleaning.csv" as input and fine tune the hyperparamters from the random forest and gradient boosting models

2_Classifier-Catboost_finetuning.ipynb: use "id_run+InputCleaning.csv" as input and fine tune the hyperparamters from Catboost

3_ClassifierComparison&Selection&Performance.ipynb: compare and train final selected model and evaluate the performance of selected model on  test dataset


------------------------------------------ Cleaning method comparison
4_1CleaningMethodsComparison_Computation.ipynb: compute the performance measurement on each of the dataset (when compared to the video observation), and output the following files: 
    "1secTS\1secTimeSeries_ALL_+str(BatchID)+.csv": 1sec categorical time series for each method. one file per batch
    "1secTimeSeries_reliability.csv": the 1sec ts over all batches (used to assess all the methods performance)
    "reliability_transition.csv" the transitions over all batches (used to assess all the methods performance)
    "ModelComparison_RESULTS.csv" with the performance measurement  

4_2CleaningMethodsComparison_Visual.ipynb: visualise the stability and the model evaluation based on the "ModelComparison_RESULTS.csv" and "reliability_transition.csv" files


------------------------------------------ Impact of cleaning on analysis
5_1Cleaning&RawdataComparison-createVariables.ipynb: clean the unprocessed records and extract variables from the cleaned data. Main output file: 'ML-method_daily_ALL_variables.csv'

5_2Cleaning&RawdataComparison-EstimatedErrorRate.ipynb: Using 'ML-method_daily_ALL_variables.csv', create visuals and compute estimated error rate

5_3Factors&estimatederror.ipynb: Estimated error rate association with environmental factors




