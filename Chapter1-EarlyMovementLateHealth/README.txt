
----------------------------------- STEP 1 ------------------------------------------
1_SelectHens&Data.ipynb
select hens (with HA5 and tracking data from day1 onward) and merge all useful data (initial weight, weather,...)
output: df_MVT_4individuality.csv


----------------------------------- STEP 2 ------------------------------------------
2_PCA.ipynb
perform PCA and save first component 
output: df_MVT_4individuality_withPCA.csv


----------------------------------- STEP 3 ------------------------------------------
3_BRMS_Predictability.ipynb
using brms packages (baysian approches) fit a double hierarchical mixed effects model and save variability estimates
output: BLUPS_variability.csv


----------------------------------- LmerApproach ------------------------------------------
LmerApproach_3_RI_RR.ipynb
use random intercept and slopes models (with comparison) to save BLUPS and daily predictions 
output: BLUP_RI.csv, BLUP_RR.csv, Prediction_RI.csv, Prediction_RR_linear.csv, Prediction_RR_quadratic.csv

LmerApproach_4_df4HA_Visual.ipynb
visualise raw with predictions from all models (RI, RR_linearslope, RR_quadratic_slope)
visualise blups
produce dataframe with all blups and first week mvt var and HA5 (80 rows), named df_MVT_4stat)
output: visuals and df_MVT_4stat.csv used in notebook named "LmerApproach_5_PC1&HA"


----------------------------------- McmcApproach ------------------------------------------
McmcApproach_4_df4HA.ipynb
from BLUPS_variability.csv and df_MVT_4individuality_withPCA.csv and daily movements (ALL_DATA_daily_ALL_variables_verified.csv),
merge the predictability BLUPS, the first week mvt variable into one dataframe (named: df_MVT_4stat_BI.csv) used in step 5 (McmcApproach_5_PC1&HA.ipynb)
output: df_MVT_4stat_BI.csv

McmcApproach_5_PC1&Feather.ipynb ; McmcApproach_5_PC1&Severity.ipynb
bivariate approach to see assocaition between mvt development variables (blups intercetp, linear slope, quadratic slope, predictability, #days with no transitions during first three days)









