#!/usr/bin/env python
# coding: utf-8

# In[135]:


import pandas as pd
#from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
import numpy as np
import os

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor


# In[121]:





df_sample = pd.read_csv("data/sample.csv")
#sample = open("data/sample.csv")
#sample = open(os.path.join(os.getcwd(),"data/sample.csv"))

df_train_feats = pd.read_csv("data/train_features.csv")
df_train_labels = pd.read_csv("data/train_labels.csv")
df_test_feats = pd.read_csv("data/test_features.csv")

df_labels_cols = list(df_train_labels)

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


def feats_2_X(df_feats,feats_list):
    #Subtask1
    n_derived_feats = 4

    #pids = np.unique(df_train_feats['pid'].values)
    pids = pd.unique(df_feats['pid'])
    n_patients = len(pids)
    #df_train_feats2 = pd.DataFrame(data={'pid':pids})

    print(n_patients)
    print(n_derived_feats*len(feats_list))
    X = np.zeros((n_patients,n_derived_feats*len(feats_list)))

    for i in range(0,n_patients):
        patient_df = df_feats[df_feats['pid']==pids[i]]
        for j in range(0,len(feats_list)):
           patient_feat = patient_df[feats_list[j]].values
           j0 = j*n_derived_feats
           X[i][j0+0] = np.mean(patient_feat)
           X[i][j0+1] = np.std(patient_feat)
           X[i][j0+2] = np.min(patient_feat)
           X[i][j0+3] = np.max(patient_feat)
        #print(patient_df)
        if i%500==0:
            print(i)
    return(X)


# In[104]:



X_train = feats_2_X(df_train_feats,feats_list2)


# In[105]:


X_test = feats_2_X(df_test_feats,feats_list2)


# In[ ]:





# In[109]:



def impute(X_train,X_test):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(X_train)
    
    X_train_imp = imp_mean.transform(X_train)
    X_test_imp = imp_mean.transform(X_test)

    #print(X_train_imp)
    return(X_train_imp,X_test_imp)


# In[107]:


print(np.shape(X_train))
print(np.shape(X_test))
X_train_imp,X_test_imp = impute(X_train,X_test)


# In[108]:


print(np.shape(X_train_imp))
print(np.shape(X_test_imp))


# In[112]:


count = 0
for i in range(np.shape(X_train)[1]):
    a = np.count_nonzero(~np.isnan(X_train[:,i]))
    print(a)
    if a>0:
        count+=1


# In[113]:


print(count)


# In[139]:


def fit_model1(X_train,y_train):
    print(np.shape(y_train))
    print(np.sum(y_train))
    clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, 
                                     max_depth=1, random_state=42).fit(X_train,y_train)
    #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, 
    #                                 max_depth=5, random_state=42).fit(X_train_imp,y_train)

    return(clf)


def fit_model3(X_train,y_train):
    #print(np.shape(y_train))
    #print(np.sum(y_train))
    reg = GradientBoostingRegressor(random_state=42).fit(X_train,y_train)

    return(reg)


# In[130]:




#print(np.shape(y_train))

test_pids = pd.unique(df_test_feats['pid'])
#y_test = np.zeros((len(test_pids),1+len(subtask1_labels)))
#y_test[:,0] = test_pids

df_test_labels = pd.DataFrame(columns=df_labels_cols)
df_test_labels['pid'] = test_pids


    
    
    


# In[ ]:


for i in range(0,len(subtask1_labels)):
#for i in range(0,1):
    y_train=df_train_labels[subtask1_labels[i]].values
    clf = fit_model1(X_train_imp,y_train)
    
    score = clf.score(X_train_imp,y_train)
    print(score)
    p_train=clf.predict_proba(X_train_imp)
    #print(p)
    print(np.sum(p_train[:,0]))
    print(np.sum(p_train[:,1]))
    
    p_test=clf.predict_proba(X_test_imp)
    print(np.sum(p_test[:,0]))
    print(np.sum(p_test[:,1]))
    
    #y_test[:,1+i] = p_test[:,1]
    
    df_test_labels[subtask1_labels[i]] = p_test[:,1]


# In[131]:





# In[134]:


for i in range(0,len(subtask2_labels)):
#for i in range(0,1):
    y_train=df_train_labels[subtask2_labels[i]].values
    clf = fit_model1(X_train_imp,y_train)
    
    score = clf.score(X_train_imp,y_train)
    print(score)
    p_train=clf.predict_proba(X_train_imp)
    #print(p)
    print(np.sum(p_train[:,0]))
    print(np.sum(p_train[:,1]))
    
    p_test=clf.predict_proba(X_test_imp)
    print(np.sum(p_test[:,0]))
    print(np.sum(p_test[:,1]))
    
    #y_test[:,1+i] = p_test[:,1]
    
    df_test_labels[subtask2_labels[i]] = p_test[:,1]


# In[142]:


for i in range(0,len(subtask3_labels)):
#for i in range(0,1):
    print("i=%d"%i)
    y_train=df_train_labels[subtask3_labels[i]].values
    reg = fit_model3(X_train_imp,y_train)
    
    score = reg.score(X_train_imp,y_train)
    print("score: %f"%score)
    
    y_train_pred=reg.predict(X_train_imp)
    print(np.mean(y_train_pred))
    print(np.std(y_train_pred))
    
    y_test=reg.predict(X_test_imp)
    print(np.mean(y_test))
    print(np.std(y_test))
    
    df_test_labels[subtask3_labels[i]] = y_test


# In[ ]:





# In[143]:


print(test_pids)
print(y_test)
print(df_test_labels)
print(np.sum(df_test_labels['LABEL_BaseExcess']))


# In[146]:


df_test_labels = df_test_labels.set_index('pid')


# In[147]:


df_test_labels


# In[148]:


df_test_labels.to_csv("submission.csv")


# In[ ]:




