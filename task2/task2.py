import pandas as pd
#from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
import numpy as np
import os


sample = pd.read_csv("data/sample.csv")
#sample = open("data/sample.csv")
#sample = open(os.path.join(os.getcwd(),"data/sample.csv"))

df_train_feats = pd.read_csv("data/train_features.csv")
df_train_labels = pd.read_csv("data/train_labels.csv")

feats_list2 = [
 'Age',
 'EtCO2',
 'PTT',
 'BUN',
 'Lactate',
 'Temp',
 'Hgb',
 'HCO3',
 'BaseExcess',
 'RRate',
 'Fibrinogen',
 'Phosphate',
 'WBC',
 'Creatinine',
 'PaCO2',
 'AST',
 'FiO2',
 'Platelets',
 'SaO2',
 'Glucose',
 'ABPm',
 'Magnesium',
 'Potassium',
 'ABPd',
 'Calcium',
 'Alkalinephos',
 'SpO2',
 'Bilirubin_direct',
 'Chloride',
 'Hct',
 'Heartrate',
 'Bilirubin_total',
 'TroponinI',
 'ABPs',
 'pH']

subtask1_labels = [
 'LABEL_BaseExcess',
 'LABEL_Fibrinogen',
 'LABEL_AST',
 'LABEL_Alkalinephos',
 'LABEL_Bilirubin_total',
 'LABEL_Lactate',
 'LABEL_TroponinI',
 'LABEL_SaO2',
 'LABEL_Bilirubin_direct',
 'LABEL_EtCO2'
        ]

subtask2_labels = [
 'LABEL_Sepsis'
 ]

subtask3_labels = [
 'LABEL_RRate',
 'LABEL_ABPm',
 'LABEL_SpO2',
 'LABEL_Heartrate'
        ]


#Subtask1
n_derived_feats = 4

#pids = np.unique(df_train_feats['pid'].values)
pids = df_train_labels['pid'].values
n_patients = len(pids)
df_train_feats2 = pd.DataFrame(data={'pid':pids})


X_train = np.zeros((n_patients,n_derived_feats*len(feats_list2)))

#for i in range(0,1):
for i in range(0,n_patients):
    patient_df = df_train_feats[df_train_feats['pid']==pids[i]]
    for j in range(0,len(feats_list2)):
       patient_feat = patient_df[feats_list2[j]].values
       j0 = j*n_derived_feats
       X_train[i][j0+0] = np.mean(patient_feat)
       X_train[i][j0+1] = np.std(patient_feat)
       X_train[i][j0+2] = np.min(patient_feat)
       X_train[i][j0+3] = np.max(patient_feat)
    #print(patient_df)
    print(i)



