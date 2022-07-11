------------------------------------------------------------------------------------------------------------------------------------------
----------------------------- Steps to perform the individual consistency and behavioural syndrome analysis ------------------------------
------------------------------------------------------------------------------------------------------------------------------------------

------------------------------ step 0 ------------------------------
0_select_hens.ipynb: get the data used for the chapter
it select hens and days based on all initial data (focal bird info, weather, movement)
output file: df_BS.csv

------------------------------ step 1 ------------------------------
1_Repeatabilities.ipynb: estimate repeatabilities
input: df_BS.csv
output: R_estimates_*.csv, with * correpsonding to one of the five behaviour

------------------------------ step 2 ------------------------------
2_Mvt_Behavioural syndrome *.ipynb: multivariate model to see existence of behavioural syndrome * can be ALLOBS: using all observations, ELS: using early life stage only, or LLS: using late life stage only. we kept three jupyter notebook althought the same, just to easily see the different output
input: df_BS.csv, but also using the result of step 1, by includding only trait that are repeatable over time and across context
output: model: ALL_Multi_DHGLM_all_var_FINAL.rda, fixed effect plot, samples from posterior distribution (not averaged: ALL_df_CI_BRMS_BS_pred_allvar_FINAL.csv but also average: ALL_df_mean_BRMS_BS_pred_allvar_FINAL.csv)

------------------------------ step 3 ------------------------------
3_Visual.ipynb: visualise results
input:  R_estimates_*.csv from step 1 and ALL_df_CI_BRMS_BS_pred_allvar_FINAL.csv & ALL_df_mean_BRMS_BS_pred_allvar_FINAL.csv from step 2









