------------------------------------------------------------------------------------------------------------------------------------------
----------------------------- Steps to perform the individual consistency and behavioural syndrome analysis ------------------------------
------------------------------------------------------------------------------------------------------------------------------------------

To run the code, each steps should be done in the below order. We used jupyter notebook so that code is divided into several paragraph which output is shown in the cells, for making it more userfriendly. Jupyter notebooks can be installed on windows following instruction, as e.g. https://www.geeksforgeeks.org/how-to-install-jupyter-notebook-in-windows/. We used R for modelling (step 1, step 2) and python for visualising (step 3).
Before runing a notebook, always look at the section "Define parameters", and replace parameters adequately to your need. Note the "Define parameters" section is either the first or second section, depending on if we need to choose subdataframe

------------------------------ step 1 ------------------------------
1_Repeatabilities.ipynb: estimate repeatabilities in R
input: df_BS.csv
output: R_estimates_*.csv, with * correpsonding to one of the five behaviour

------------------------------ step 2 ------------------------------
2_Mvt_Behavioural syndrome-ALLOBS.ipynb: multivariate model to see existence of behavioural syndrome in R
input: df_BS.csv, but also using the result of step 1, by includding only trait that are repeatable over time and across context
output: model: ALL_Multi_DHGLM_all_var_FINAL.rda, fixed effect plot, samples from posterior distribution (not averaged: ALL_df_CI_BRMS_BS_pred_allvar_FINAL.csv but also average: ALL_df_mean_BRMS_BS_pred_allvar_FINAL.csv)

------------------------------ step 3 ------------------------------
3_Visual.ipynb: visualise results in python
input:  R_estimates_*.csv from step 1 and ALL_df_CI_BRMS_BS_pred_allvar_FINAL.csv & ALL_df_mean_BRMS_BS_pred_allvar_FINAL.csv from step 2
output: visuals and "BS_lm_metadata.csv" used to do a simple linear model in R 










