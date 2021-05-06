

------------------------------------------ Decision tree classifiers
1_PreprocessedTraining&TestingData.ipynb: merge the records metadata with the video analysis. output: "id_run+InputCleaning.csv"

2_Classifier-RF&GradientBoosting_finetuning.ipynb: use "id_run+InputCleaning.csv" as input and fine tune the hyperparamters from the random forest and gradient boosting models

2_Classifier-Catboost_finetuning.ipynb: use "id_run+InputCleaning.csv" as input and fine tune the hyperparamters from Catboost

3_ClassifierComparison&Selection&Performance.ipynb: compare and train final selected model and evaluate the performance of selected model


------------------------------------------ Cleaning method comparison
4_1CleaningMethodsComparison_Computation.ipynb: compute the performance measurement on each of the frou dataset (when compared to the video observation), and output the following files: 
    "1secTS\1secTimeSeries_ALL_+str(BatchID)+.csv": 1sec categorical time series for each method. one file per batch
    "1secTimeSeries_reliability.csv": the 1sec ts over all batches (used to assess all the methods performance)
    "reliability_transition.csv" the transitions over all batches (used to assess all the methods performance)
    "ModelComparison_RESULTS.csv" with the performance measurement  

4_2CleaningMethodsComparison_Visual.ipynb: visualise the stability and the model evaluation based on the "ModelComparison_RESULTS.csv" and "reliability_transition.csv" files


------------------------------------------ Impact of cleaning on analysis
5_1Cleaning&RawdataComparison-createVariables.ipynb: clean the unprocessed records and extract variables from the cleaned data. Main output file: 'ML-method_daily_ALL_variables.csv'

5_2Cleaning&RawdataComparison.ipynb: Using 'ML-method_daily_ALL_variables.csv', create visuals and compute measures to asess teh difference between unprocessed and cleaned dataset




