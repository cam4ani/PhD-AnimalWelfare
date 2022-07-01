------------------------------------------------------------------------------------------------------------------------------------------
----------------------------- Steps to perform the individual consistency and behavioural syndrome analysis ------------------------------
------------------------------------------------------------------------------------------------------------------------------------------

step 0: get the data used for this paper, using the notebook: 0_select_hens.ipynb
it select hens and days based on all initial data (focal bird info, weather, movement)
output file: df_BS.csv

step 1: estimate repeatabilities, using the notebook: 1_Repeatabilities.ipynb
input: df_BS.csv
output: R_estimates_*.csv, with * correpsonding to one of the five behaviour

step 2: multivariate model to see existence of behavioural syndrome using notebook: 2_Mvt_Behavioural syndrome.ipynb
input: df_BS.csv, but also using the result of step 1, using only trait that are repeatable over time and across context
output: model: ALL_Multi_DHGLM_all_var_FINAL.rda, fixed effect  plot, samples from posterior distribution not averaged: ALL_df_CI_BRMS_BS_pred_allvar_FINAL.csv but also average: ALL_df_mean_BRMS_BS_pred_allvar_FINAL.csv

step 3: visualise results, using the notebook: 3_Visual.ipynb
input:  R_estimates_*.csv fromm step 1 and ALL_df_CI_BRMS_BS_pred_allvar_FINAL.csv & ALL_df_mean_BRMS_BS_pred_allvar_FINAL.csv from step 2









