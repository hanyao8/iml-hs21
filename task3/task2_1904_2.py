#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


import numpy as np
import os

import sklearn.metrics as metrics

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, recall_score,accuracy_score, precision_score, r2_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import sklearn
print(sklearn.__version__)


# In[24]:


holdout=False


# In[3]:


df_sample = pd.read_csv("data/sample.csv")

df_train_feats = pd.read_csv("data/train_features.csv")
df_train_labels = pd.read_csv("data/train_labels.csv")
df_test_feats = pd.read_csv("data/test_features.csv")

df_labels_cols = list(df_train_labels)

active_feats = [
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

subtask3_feats = [
 'RRate',
 'ABPm',
 'SpO2',
 'Heartrate']

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


def feats_2_X1(df_feats,feats_list):
    #Subtask1
    n_derived_feats = 4

    pids = pd.unique(df_feats['pid'])
    n_patients = len(pids)

    print(n_patients)
    print(n_derived_feats*len(feats_list))
    X = np.nan*np.ones((n_patients,n_derived_feats*len(feats_list)))

    patient_df_sizes = np.zeros(n_patients)
    for i in range(0,n_patients):
        patient_df = df_feats[df_feats['pid']==pids[i]]
        patient_df_sizes[i] = patient_df.shape[0]
        for j in range(0,len(feats_list)):
            patient_feat = patient_df[feats_list[j]].values
            patient_feat = patient_feat[~np.isnan(patient_feat)]
            #print(len(patient_feat))
            j0 = j*n_derived_feats
            
            if len(patient_feat)>0:
                X[i][j0+0] = np.mean(patient_feat)
                X[i][j0+1] = np.std(patient_feat)
                X[i][j0+2] = np.min(patient_feat)
                X[i][j0+3] = np.max(patient_feat)

        if i%1000==0:
            print(i)
    print(np.max(patient_df_sizes))
    return(X)

def feats_2_X2(df_feats,feats_list):
    #Subtask2
    n_derived_feats = 4
    #n_derived_feats = 1

    pids = pd.unique(df_feats['pid'])
    n_patients = len(pids)

    print(n_patients)
    print(n_derived_feats*len(feats_list))
    X = np.nan*np.ones((n_patients,n_derived_feats*len(feats_list)))

    patient_df_sizes = np.zeros(n_patients)
    for i in range(0,n_patients):
        patient_df = df_feats[df_feats['pid']==pids[i]]
        patient_df_sizes[i] = patient_df.shape[0]
        for j in range(0,len(feats_list)):
            patient_feat = patient_df[feats_list[j]].values
            patient_feat = patient_feat[~np.isnan(patient_feat)]
            #print(len(patient_feat))
            j0 = j*n_derived_feats
            
            if len(patient_feat)>0:
                X[i][j0+0] = np.mean(patient_feat)
                X[i][j0+1] = np.std(patient_feat)
                X[i][j0+2] = np.min(patient_feat)
                X[i][j0+3] = np.max(patient_feat)

        if i%1000==0:
            print(i)
    print(np.max(patient_df_sizes))
    return(X)


def feats_2_X3(df_feats,feats_list):
    #Subtask3
    n_derived_feats = 8

    pids = pd.unique(df_feats['pid'])
    n_patients = len(pids)

    print(n_patients)
    print(n_derived_feats*len(feats_list))
    X = np.nan*np.ones((n_patients,n_derived_feats*len(feats_list)))

    patient_df_sizes = np.zeros(n_patients)
    for i in range(0,n_patients):
        patient_df = df_feats[df_feats['pid']==pids[i]]
        patient_df_sizes[i] = patient_df.shape[0]
        for j in range(0,len(feats_list)):
            patient_feat = patient_df[feats_list[j]].values
            nan_mask = np.isnan(patient_feat)
            times = (patient_df['Time'].values)[~nan_mask]
            times = times.astype('float')
            patient_feat = patient_feat[~nan_mask]
            #print(len(patient_feat))
            j0 = j*n_derived_feats
            
            n_measurements = len(patient_feat)
            if n_measurements>0:
                X[i][j0+0] = np.mean(patient_feat)
            if n_measurements>1:
                X[i][j0+1] = patient_feat[-1]
                grad_0 = (patient_feat[-1]-patient_feat[0])/(times[-1]-times[0])
                X[i][j0+2] = grad_0
            if n_measurements>2:
                mid_idx = int((n_measurements-1)/2)
                X[i][j0+3] = patient_feat[mid_idx]
                
                grad_1 = (patient_feat[-1]-patient_feat[-2])/(times[-1]-times[-2])
                X[i][j0+4] = grad_1
                grad_2 = (patient_feat[-2]-patient_feat[-3])/(times[-2]-times[-3])
                X[i][j0+5] = grad_2
                
                curve_0 = (grad_1-grad_0)/(0.5*times[-2]-0.5*times[0])
                X[i][j0+6] = curve_0
                curve_1 = (grad_1-grad_2)/(0.5*times[-1]-0.5*times[-3])
                X[i][j0+7] = curve_1
                
        if i%1000==0:
            print(i)
    print(np.max(patient_df_sizes))
    return(X)


# In[4]:


def standardize(X_train,X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return(X_train,X_test)
    

def impute(X_train,X_test):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(X_train)
    
    X_train = imp_mean.transform(X_train)
    X_test = imp_mean.transform(X_test)

    #print(X_train_imp)
    return(X_train,X_test)

def forest_fi(X_train,y_train,X_test):

    forest = ExtraTreesClassifier(n_estimators=20,
                                  random_state=0)

    forest.fit(X_train,y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    X_train = X_train[:,indices[:50]]
    X_test = X_test[:,indices[:50]]
    return(X_train,X_test)


def nystroem(X_train,X_test):
    gamma=1.0
    n_components=100
    print("nystroem gamma=%f"%(gamma))
    print("nystroem q=%d"%(n_components))
    feature_map_nystroem = Nystroem(gamma=gamma,
                                    random_state=42,
                                    n_components=n_components)
    feature_map_nystroem.fit(X_train)
    Q_train = feature_map_nystroem.transform(X_train)
    sqrt_k_inv_train = np.linalg.inv(feature_map_nystroem.normalization_)
    B_train = np.dot(Q_train,sqrt_k_inv_train)
    K_train = np.dot(B_train,np.transpose(B_train))

    Q_test = feature_map_nystroem.transform(X_test)
    sqrt_k_inv_test = np.linalg.inv(feature_map_nystroem.normalization_)
    B_test = np.dot(Q_test,sqrt_k_inv_test)
    K_test = np.dot(B_test,np.transpose(B_train))
    return(K_train,K_test)


# In[5]:


def sigmoid(x):
    return(1/(1+np.exp(-x)))

def fit_model1_1(X_train,y_train):
    print(np.shape(y_train))
    print(np.sum(y_train))
    clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, 
                                     max_depth=3, random_state=42).fit(X_train,y_train)
    #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, 
    #                                 max_depth=5, random_state=42).fit(X_train_imp,y_train)

    return(clf)

def fit_model1_2(X_train,y_train):
    print(np.shape(y_train))
    print(np.sum(y_train))
    
    clf = svm.SVC(C=10,
                  class_weight="balanced",
                  decision_function_shape='ovo').fit(X_train,y_train)
    return(clf)

def fit_model1(X_train,y_train,clf_init,sw_dict={}):
    n = np.shape(y_train)[0]
    w0 = n/(n-np.sum(y_train))
    w1 = n/np.sum(y_train)
    
    sample_weight = np.zeros(len(y_train))
    if not(sw_dict):
        print("using default sample weights")
        sample_weight[y_train == 0] = w0
        sample_weight[y_train == 1] = w1
    else:
        print("using custom sample weights")
        sample_weight[y_train == 0] = sw_dict[0]
        sample_weight[y_train == 1] = sw_dict[1]
    
    print("Shape and sum of y_train")
    print(np.shape(y_train))
    print(np.sum(y_train))
    clf = clf_init
    #clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, 
    #                                 max_depth=3, random_state=42)
    clf.fit(X_train,y_train,sample_weight=sample_weight)
    #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, 
    #                                 max_depth=5, random_state=42).fit(X_train_imp,y_train)

    return(clf)

def fit_model2_1(X_train,y_train,hps):
    C = hps["C"]
    #class_weight={1:16.5}
    class_weight={1:hps["w1"]}
    print("Shape and sum of y_train")
    print(np.shape(y_train))
    print(np.sum(y_train))
    
    clf = svm.SVC(C=hps["C"],
                  kernel="rbf",
                  gamma=hps["gamma"],
                  class_weight=class_weight,
                  decision_function_shape='ovo',
                  verbose=True).fit(X_train,y_train)
    print("C: %f"%hps["C"])
    print("gamma: %f"%hps["gamma"])
    print("Computed class weight: %s"%(str(clf.class_weight_)))
    return(clf)

def fit_model2_2(K_train,y_train):
    C = 1.0
    class_weight={1:17.5}
    print("Shape and sum of y_train")
    print(np.shape(y_train))
    print(np.sum(y_train))
    
    clf = svm.SVC(C=C,
                  kernel="precomputed",
                  class_weight="balanced",
                  decision_function_shape='ovo',
                  verbose=True).fit(K_train,y_train)
    print("C: %f"%C)
    print("Computed class weight: %s"%(str(clf.class_weight_)))
    return(clf)

def fit_model2(X_train,y_train,clf_init,sw_dict={}):
    n = np.shape(y_train)[0]
    w0 = n/(n-np.sum(y_train))
    w1 = n/np.sum(y_train)
    
    sample_weight = np.zeros(len(y_train))
    if not(sw_dict):
        print("using default sample weights")
        sample_weight[y_train == 0] = w0
        sample_weight[y_train == 1] = w1
    else:
        print("using custom sample weights")
        sample_weight[y_train == 0] = sw_dict[0]
        sample_weight[y_train == 1] = sw_dict[1]
    
    print("Shape and sum of y_train")
    print(np.shape(y_train))
    print(np.sum(y_train))
    clf = clf_init
    clf.fit(X_train,y_train,sample_weight=sample_weight)
    return(clf)

def fit_model3(X_train,y_train):
    #print(np.shape(y_train))
    #print(np.sum(y_train))
    reg = HistGradientBoostingRegressor(random_state=42).fit(X_train,y_train)
    #reg = GradientBoostingRegressor(random_state=42).fit(X_train,y_train)
    return(reg)


# In[6]:


test_pids = pd.unique(df_test_feats['pid'])

df_test_labels = pd.DataFrame(columns=df_labels_cols)
df_test_labels['pid'] = test_pids


# In[7]:


X_train1_raw = feats_2_X1(df_train_feats,active_feats)
X_test1_raw = feats_2_X1(df_test_feats,active_feats)


# In[26]:


print(np.shape(X_train1_raw))
print(np.shape(X_test1_raw))
X_train_imp,X_test = impute(X_train1_raw,X_test1_raw)
print(np.shape(X_train_imp))
print(np.shape(X_test))

holdout_aucs = np.zeros(len(subtask1_labels))

for i in range(0,len(subtask1_labels)):
#for i in range(0,1):
    print("i=%d"%i)
    y_train=df_train_labels[subtask1_labels[i]].values
    n = float(np.shape(y_train)[0])

    if holdout:
        #make hold out
        X_train,X_hold,y_train,y_hold = train_test_split(
            X_train_imp,y_train,test_size=0.2,random_state=42)

        print(np.shape(X_train))
        print(np.shape(X_hold))
        print(np.sum(y_train))
        print(np.sum(y_hold))
        print("\n")
    else:
        X_train = X_train_imp

    #sample weight
    use_custom_sw = True
    if use_custom_sw:
        #sample weight
        n = float(np.shape(y_train)[0])
        w1_boost = 0.17

        w0 = n/(n-np.sum(y_train))
        w1 = (n/np.sum(y_train))*w1_boost
        geo_mean = np.sqrt(w0*w1)
        w0 /= geo_mean
        w1 /= geo_mean
    else:
        w0=1.0
        w1=1.0
    
    clf_init = HistGradientBoostingClassifier(random_state=42)
    clf = fit_model1(X_train,y_train,clf_init,sw_dict={0:w0,1:w1})
    score = clf.score(X_train,y_train)
    
    cv_results = cross_validate(clf,X_train,y_train,cv=5,
            scoring=["roc_auc","accuracy","recall","precision"])
    
    print("train score= %f"%score)
    print("cv roc auc= %f"%np.mean(cv_results['test_roc_auc']))
    print("cv acc= %f"%np.mean(cv_results['test_accuracy']))
    print("cv rec= %f"%np.mean(cv_results['test_recall']))
    print("cv prec= %f"%np.mean(cv_results['test_precision']))
    
    #print(sigmoid(clf.decision_function(X_train)))
    #p_train=clf.predict(X_train)
    
    print("y train:")
    print("n pos %f"%np.sum(y_train))
    print("frac pos %f"%(np.sum(y_train)/n))
    
    print("y train pred:")
    y_train_pred=clf.predict(X_train)
    print("n pos %f"%np.sum(y_train_pred))
    print("frac pos %f"%(np.sum(y_train_pred)/n) ) 
    
    p_train=clf.predict_proba(X_train)
    print("prob sum 0 %f"%np.sum(p_train[:,0]))
    print("prob sum 1 %f"%np.sum(p_train[:,1]))
    
    #print(p_train)
    
    print("y test:")
    p_test=clf.predict_proba(X_test)
    print("prob sum 0 %f"%np.sum(p_test[:,0]))
    print("prob sum 1 %f"%np.sum(p_test[:,1]))
    
    if holdout:
        print("holdout metrics:")
        holdout_auc = roc_auc_score(y_hold,clf.predict_proba(X_hold)[:,1])
        print("holdout roc auc= %f"%holdout_auc)
        holdout_aucs[i] = holdout_auc

        y_hold_pred = clf.predict(X_hold)
        holdout_acc = accuracy_score(y_hold,y_hold_pred)
        print("holdout acc= %f"%holdout_acc)
        holdout_rec = recall_score(y_hold,y_hold_pred)
        print("holdout rec= %f"%holdout_rec)
        holdout_prec = precision_score(y_hold,y_hold_pred)
        print("holdout prec= %f"%holdout_prec)
    
    print("\n\n")
    
    #y_test[:,1+i] = p_test[:,1]
    
    df_test_labels[subtask1_labels[i]] = p_test[:,1]
    
print("Hold out aucs and mean")
print(holdout_aucs)
print(np.mean(holdout_aucs))


# In[10]:


X_train2_raw = feats_2_X2(df_train_feats,active_feats)
X_test2_raw = feats_2_X2(df_test_feats,active_feats)



# In[12]:


print(np.shape(X_train2_raw))
print(np.shape(X_test2_raw))

X_train_imp,X_test = impute(X_train2_raw,X_test2_raw)
print(np.shape(X_train))
print(np.shape(X_test))

#X_train,X_test = standardize(X_train,X_test)
#print(np.shape(X_train))
#print(np.shape(X_test))


#X_train,X_test = nystroem(X_train,X_test)
#print(np.shape(X_train))
#print(np.shape(X_test))

hp_x = np.array([0.05])
hp_metrics = ['hold_roc_auc',
           'hold_accuracy',
           'hold_recall',
           'hold_precision'
          ]
hp_y = np.zeros((hp_x.shape[0],len(hp_metrics)))

for j in range(0,hp_x.shape[0]):
    for i in range(0,len(subtask2_labels)):
    #for i in range(0,1):
        print("i=%d"%i)
        y_train=df_train_labels[subtask2_labels[i]].values
        n = float(np.shape(y_train)[0])

        if holdout:
            #make hold out
            X_train,X_hold,y_train,y_hold = train_test_split(
                X_train_imp,y_train,test_size=0.2,random_state=42)

            print(np.shape(X_train))
            print(np.shape(X_hold))
            print(np.sum(y_train))
            print(np.sum(y_hold))
            print("\n")
        else:
            X_train = X_train_imp

        #sample weight
        use_custom_sw = True
        if use_custom_sw:
            #sample weight
            n = float(np.shape(y_train)[0])
            w1_boost = 0.17

            w0 = n/(n-np.sum(y_train))
            w1 = (n/np.sum(y_train))*w1_boost
            geo_mean = np.sqrt(w0*w1)
            w0 /= geo_mean
            w1 /= geo_mean
        else:
            w0=1.0
            w1=1.0
            
        print("class weights: %f,%f"%(w0,w1))

        clf_init = HistGradientBoostingClassifier(
            learning_rate=hp_x[j],random_state=42)
        clf = fit_model2(X_train,y_train,clf_init,sw_dict={0:w0,1:w1})
        score = clf.score(X_train,y_train)

        cv_results = cross_validate(clf,X_train,y_train,cv=5,
                scoring=["roc_auc","accuracy","recall","precision"])

        print("train score= %f"%score)
        print("cv roc auc= %f"%np.mean(cv_results['test_roc_auc']))
        print("cv acc= %f"%np.mean(cv_results['test_accuracy']))
        print("cv rec= %f"%np.mean(cv_results['test_recall']))
        print("cv prec= %f"%np.mean(cv_results['test_precision']))

        #print(sigmoid(clf.decision_function(X_train)))
        #p_train=clf.predict(X_train)

        print("y train:")
        print("n pos %f"%np.sum(y_train))
        print("frac pos %f"%(np.sum(y_train)/n))

        print("y train pred:")
        y_train_pred=clf.predict(X_train)
        print("n pos %f"%np.sum(y_train_pred))
        print("frac pos %f"%(np.sum(y_train_pred)/n) ) 

        p_train=clf.predict_proba(X_train)
        print("prob sum 0 %f"%np.sum(p_train[:,0]))
        print("prob sum 1 %f"%np.sum(p_train[:,1]))

        #print(p_train)

        print("y test:")
        p_test=clf.predict_proba(X_test)
        print("prob sum 0 %f"%np.sum(p_test[:,0]))
        print("prob sum 1 %f"%np.sum(p_test[:,1]))

        if holdout:

            print("holdout metrics:")
            holdout_auc = roc_auc_score(y_hold,clf.predict_proba(X_hold)[:,1])
            print("holdout roc auc= %f"%holdout_auc)
            hp_y[j][0] = holdout_auc

            y_hold_pred = clf.predict(X_hold)
            holdout_acc = accuracy_score(y_hold,y_hold_pred)
            print("holdout acc= %f"%holdout_acc)
            hp_y[j][1] = holdout_acc

            holdout_rec = recall_score(y_hold,y_hold_pred)
            print("holdout rec= %f"%holdout_rec)
            hp_y[j][2] = holdout_rec

            holdout_prec = precision_score(y_hold,y_hold_pred)
            print("holdout prec= %f"%holdout_prec)
            hp_y[j][3] = holdout_prec


        print("\n\n")

        #y_test[:,1+i] = p_test[:,1]

        df_test_labels[subtask2_labels[i]] = p_test[:,1]


# In[13]:


f1 = plt.figure()
ax1 = f1.add_subplot(111)
for k in range(0,len(hp_metrics)):
    ax1.plot(np.log10(hp_x),hp_y[:,k],label=hp_metrics[k])
ax1.legend()
plt.show()


# In[14]:


df_test_labels


# In[15]:


X_train3_raw = feats_2_X3(df_train_feats,subtask3_feats)
X_test3_raw = feats_2_X3(df_test_feats,subtask3_feats)


# In[17]:


print(np.shape(X_train3_raw))
print(np.shape(X_test3_raw))

X_train_imp,X_test = impute(X_train3_raw,X_test3_raw)
print(np.shape(X_train_imp))
print(np.shape(X_test))

#X_train,X_test = standardize(X_train,X_test)
#print(np.shape(X_train))
#print(np.shape(X_test))



holdout_r2s = np.zeros(len(subtask3_labels))

for i in range(0,len(subtask3_labels)):
#for i in range(0,1):
    print("i=%d"%i)
    y_train=df_train_labels[subtask3_labels[i]].values
    
    if holdout:
        #make hold out
        X_train,X_hold,y_train,y_hold = train_test_split(
            X_train_imp,y_train,test_size=0.2,random_state=42)

        print(np.shape(X_train))
        print(np.shape(X_hold))
        print(np.sum(y_train))
        print(np.sum(y_hold))
        print("\n")
    else:
        X_train = X_train_imp

    reg = fit_model3(X_train,y_train)
    
    score = reg.score(X_train,y_train)
    cv_results = cross_validate(reg,X_train,y_train,cv=5,
            scoring=["r2"])
    
    print("train score= %f"%score)
    print("cv r2= %f"%np.mean(cv_results['test_r2']))
    
    y_train_pred=reg.predict(X_train)
    print(np.mean(y_train_pred))
    print(np.std(y_train_pred))
    
    y_test=reg.predict(X_test)
    print(np.mean(y_test))
    print(np.std(y_test))
    
    if holdout:
        print("holdout metrics:")
        holdout_r2 = r2_score(y_hold,reg.predict(X_hold))
        print("holdout roc auc= %f"%holdout_r2)
        holdout_r2s[i] = holdout_r2
        #hp_y[j][0] = holdout_auc
    
    df_test_labels[subtask3_labels[i]] = y_test


# In[22]:


df_test_labels


# In[27]:


df_test_labels.to_csv('prediction_1904_8.zip', index=False, float_format='%.3f', compression='zip')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




