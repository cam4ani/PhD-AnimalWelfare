
BY STORIES

------------------TPM: topic modelling for behavioral understanding
Purpose
describe the evolution of movement. Can also induce MLPSIDS

Notebook
TPM_1_topic_modelling_allsession: topic modelling with LDA (latent dirichelet models) on the entire period


------------------TPMLDA: topic modelling and LDA on the same period to predict future health/welfare
Purpose
Apply topic modelling on x consecutives days, then LDA (class: HenID, nbr observation per class:x) to see how these variables predict future health/welfare issues that are still not out now
This analysis can be done with fix x on windowed period
also, x coul be around 20, it should not be to high: car then topic modeling will capture the MLP change induce by the age, and it hould not be to small in order to have enough obersvation per class for the LDA, and also to have enough documents in the topic modelling)

Notebooks
TPMLDA_1_topic_modelling: create models for each k
    TPMLDA_2_LDA: LDA on the topic proba
        TPMLDA_3_csv4predictingKBF: dataframe for modelling with all var
            TPMLDA_4_featureSelection: correlation graph for feature selection as not much variable! 
                TPMLDA_5_predictingKBF: power of model for max severity with all var
            
                   
        
------------------PERSONALITYTRAIT: LDA for personality trait
PERSONALITYTRAIT_Mobility-topicModelling&LDA_modelononesession_predictonall























