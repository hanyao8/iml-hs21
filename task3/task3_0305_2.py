#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

import sklearn.metrics as metrics

from sklearn import svm

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, recall_score,     accuracy_score, precision_score, r2_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

import matplotlib.pyplot as plt

import sklearn
print(sklearn.__version__)


# In[3]:


df_sample = pd.read_csv("data/sample.csv")

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")


# In[6]:


def gen_feats_dict(df_train):
    fv = df_train['Sequence'].values
    

    fv_1plex_list = []
    for i in range(len(fv)):
        for j in range(0,4):
            fv_1plex_list.append(fv[i][j:j+1])
    feats1 = list(set(fv_1plex_list))
    #print(i)
    #print(j)
    #print(len(feats1))
    
    feats_dict = {}
    
    feats_dict['1plex_pos0']=feats1
    feats_dict['1plex_pos1']=feats1
    feats_dict['1plex_pos2']=feats1
    feats_dict['1plex_pos3']=feats1
        
    return(feats_dict)


# In[7]:


def gen_occ(df,feats_dict):
    
    fv = df['Sequence'].values
    
    occ = {}
    
    fv_1plex_list = []
    for i in range(len(fv)):
        for j in range(0,4):
            fv_1plex_list.append(fv[i][j:j+1])
    #1-plex
    for i in range(len(feats_dict['1plex'])):
        feat = feats_dict['1plex'][i]
        count = fv_1plex_list.count(feat)
        occ[feat] = count
        print("%s: %d"%(feat,count))
    
    return(occ)


# In[8]:


def gen_X(df,feats_dict):
    n = df.shape[0]
    d = 0
    for feat_type in list(feats_dict):
        d+=len(feats_dict[feat_type])
    X = np.zeros((n,d))
    print(np.shape(X))
    
    seqs = df["Sequence"].values
    i=0
    for feat_type in list(feats_dict):
        pos = int(feat_type[-1])
        plex = int(feat_type[0])
        for j in range(len(feats_dict[feat_type])):

            for k in range(len(seqs)):
                if feats_dict[feat_type][j] == seqs[k][pos:pos+plex]:
                    X[k][i] = 1

            if i%10==0:
                print(i)
            i+=1

    return(X)
    


# In[9]:


feats_dict = gen_feats_dict(df_train)


# In[10]:


X_train_raw = gen_X(df_train,feats_dict)
X_test_raw = gen_X(df_test,feats_dict)


# In[12]:




holdout=False
feat_sel1=False
feat_sel2=False

if holdout:
    #make hold out
    X_train,X_hold,y_train,y_hold = train_test_split(
        X_train_raw,y_train,test_size=0.2,random_state=42)

    print(np.shape(X_train))
    print(np.shape(X_hold))
    print(np.sum(y_train))
    print(np.sum(y_hold))
    print("\n")
else:
    X_train = X_train_raw.copy()
    
X_test = X_test_raw.copy()
    
y_train = df_train['Active'].values
n = float(np.shape(y_train)[0])    

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

if feat_sel1:
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    
if feat_sel2:
    #feature selection by conditional probability threshold
    a = np.sum(X_train,axis=0)
    b = np.sum(np.transpose(X_train)*y_train,axis=1)
    print(np.shape(a))
    print(np.shape(b))
    c = b/a
    d = np.argwhere(c>0.2).flatten()
    
    X_train = X_train[:,d]
    X_test = X_test[:,d]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
    






# In[ ]:



hp_x = np.array([0.56])
hp_metrics = [
           'test_accuracy',
           'test_recall',
           'test_precision'
          ]
hp_y = np.zeros((hp_x.shape[0],len(hp_metrics)))

for j in range(0,hp_x.shape[0]):
    
    clf_init = HistGradientBoostingClassifier(
        learning_rate = hp_x[j],
        max_iter=200,
        max_leaf_nodes=127,
        min_samples_leaf=20,
        l2_regularization=3.2,
        verbose=10,
        random_state=42)
    
    clf = clf_init.fit(X_train,y_train)
    score = clf.score(X_train,y_train)

    cv_results = cross_validate(clf,X_train,y_train,cv=5,
            scoring=["roc_auc","accuracy","recall","precision"])

    print("train score= %f"%score)
    print("cv roc auc= %f"%np.mean(cv_results['test_roc_auc']))
    print("cv acc= %f"%np.mean(cv_results['test_accuracy']))
    hp_y[j][0] = np.mean(cv_results['test_accuracy'])
    
    print("cv rec= %f"%np.mean(cv_results['test_recall']))
    hp_y[j][1] = np.mean(cv_results['test_recall'])
    
    print("cv prec= %f"%np.mean(cv_results['test_precision']))
    hp_y[j][2] = np.mean(cv_results['test_precision'])

    #print(sigmoid(clf.decision_function(X_train)))
    #p_train=clf.predict(X_train)

    print("y train:")
    print("n pos %f"%np.sum(y_train))
    print("frac pos %f"%(np.sum(y_train)/n))

    print("y train pred:")
    y_train_pred=clf.predict(X_train)
    print("n pos %f"%np.sum(y_train_pred))
    print("frac pos %f"%(np.sum(y_train_pred)/n) ) 

    #p_train=clf.predict_proba(X_train)
    #print("prob sum 0 %f"%np.sum(p_train[:,0]))
    #print("prob sum 1 %f"%np.sum(p_train[:,1]))

    #print(p_train)

    #print("y test:")
    #p_test=clf.predict_proba(X_test)
    #print("prob sum 0 %f"%np.sum(p_test[:,0]))
    #print("prob sum 1 %f"%np.sum(p_test[:,1]))

    if holdout:
        #print("holdout metrics:")
        #holdout_auc = roc_auc_score(y_hold,clf.predict_proba(X_hold)[:,1])
        #print("holdout roc auc= %f"%holdout_auc)
        #holdout_aucs[i] = holdout_auc

        y_hold_pred = clf.predict(X_hold)
        holdout_acc = accuracy_score(y_hold,y_hold_pred)
        print("holdout acc= %f"%holdout_acc)
        holdout_rec = recall_score(y_hold,y_hold_pred)
        print("holdout rec= %f"%holdout_rec)
        holdout_prec = precision_score(y_hold,y_hold_pred)
        print("holdout prec= %f"%holdout_prec)


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)


# In[ ]:


p_train=clf.predict_proba(X_train)
print("prob sum 0 %f"%np.sum(p_train[:,0]))
print("prob sum 1 %f"%np.sum(p_train[:,1]))


# In[ ]:


print(n/(2*np.bincount(y_train)))


# In[ ]:




print("y test pred:")
y_test_pred=clf.predict(X_test)
print("n pos %f"%np.sum(y_test_pred))
print("frac pos %f"%(np.sum(y_test_pred)/n) )


# In[ ]:


print(y_test_pred)


# In[ ]:


result_str = ""
for i in range(len(y_test_pred)):
    result_str+=str(y_test_pred[i])
    result_str+="\n"

submission = open("sub2.csv","w")
submission.write(result_str)
submission.close()


# In[ ]:


#HIST GBDT l2, stepsize experiment, 200 trees/iter

print(hp_x)
print(hp_y)


# In[ ]:


f1 = plt.figure()
ax1 = f1.add_subplot(111)
for k in range(0,len(hp_metrics)):
    ax1.plot(np.log10(hp_x),hp_y[:,k],label=hp_metrics[k])
    
f1 = 2*hp_y[:,1]*hp_y[:,2]/(hp_y[:,1]+hp_y[:,2])
ax1.plot(np.log10(hp_x),f1,label="f1")
    
ax1.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




