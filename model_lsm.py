#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import string
import pandas as pd
import numpy as np
import datetime
import os
import pandas as pd
from collections import Counter


# In[2]:


from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
plt.rc("font", size=14)


# In[3]:


df_all = pd.read_csv(r'data/training/df_essays_processed_LIWC Analysis.csv', encoding='utf-8', low_memory = False)

df_all.head()


# In[4]:


features_for_modeling = list(range(8, df_all.shape[1]))
#features_for_modeling
evaluation = []


# In[5]:


df_all_train = df_all.iloc[:,features_for_modeling]
df_all_train.head()


# In[6]:


df_all_target_cEXT = df_all.iloc[:,[1]]
df_all_target_cEXT.head()


# In[7]:


df_all_target_cNEU = df_all.iloc[:,[2]]
df_all_target_cNEU.head()


# In[8]:


df_all_target_cAGR = df_all.iloc[:,[3]]
df_all_target_cAGR.head()


# In[9]:


df_all_target_cCON = df_all.iloc[:,[4]]
df_all_target_cCON.head()


# In[10]:


df_all_target_cOPN = df_all.iloc[:,[5]]
df_all_target_cOPN.head()


# SVM Model with linear kernel

# In[11]:


Train_X, Test_X, Train_YEXT, Test_YEXT = model_selection.train_test_split(df_all_train, df_all_target_cEXT, test_size=0.2, random_state=42)
Train_X, Test_X, Train_YNEU, Test_YNEU = model_selection.train_test_split(df_all_train, df_all_target_cNEU, test_size=0.2, random_state=42)
Train_X, Test_X, Train_YAGR, Test_YAGR = model_selection.train_test_split(df_all_train, df_all_target_cAGR, test_size=0.2, random_state=42)
Train_X, Test_X, Train_YCON, Test_YCON = model_selection.train_test_split(df_all_train, df_all_target_cCON, test_size=0.2, random_state=42)
Train_X, Test_X, Train_YOPN, Test_YOPN = model_selection.train_test_split(df_all_train, df_all_target_cOPN, test_size=0.2, random_state=42)

len(Train_X)


# In[12]:


svclassifierEXT = SVC(kernel='linear')
print("training Extraversion cEXT using SVM...")
svclassifierEXT.fit(Train_X, Train_YEXT)
evaluation.append(svclassifierEXT.score(Test_X, Test_YEXT))
print("cEXT score: ", svclassifierEXT.score(Test_X, Test_YEXT))

svclassifierNEU = SVC(kernel='linear')
print("training Neuroticism cNEU using SVM...")
svclassifierNEU.fit(Train_X, Train_YNEU)
evaluation.append(svclassifierNEU.score(Test_X, Test_YNEU))
print("cNEU score: ", svclassifierNEU.score(Test_X, Test_YNEU))

svclassifierAGR = SVC(kernel='linear')
print("training Agreeableness cAGR using SVM...")
svclassifierAGR.fit(Train_X, Train_YAGR)
evaluation.append(svclassifierAGR.score(Test_X, Test_YAGR))
print("cAGR score: ", svclassifierAGR.score(Test_X, Test_YAGR))

svclassifierCON = SVC(kernel='linear')
print("training Conscientiousness cCON using SVM...")
svclassifierCON.fit(Train_X, Train_YCON)
evaluation.append(svclassifierCON.score(Test_X, Test_YCON))
print("cCON score: ", svclassifierCON.score(Test_X, Test_YCON))

svclassifierOPN = SVC(kernel='linear')
print("training Openness to Experience cOPN using SVM...")
svclassifierOPN.fit(Train_X, Train_YOPN)
evaluation.append(svclassifierOPN.score(Test_X, Test_YOPN))
print("cOPN score: ", svclassifierOPN.score(Test_X, Test_YOPN))


# Logistic Regression

# In[13]:


from sklearn.linear_model import LogisticRegression
name="logR"

print("training Extraversion cEXT using Logistic Regression...")
clf_log_cEXT = LogisticRegression(solver="newton-cg")
clf_log_cEXT.fit(Train_X, Train_YEXT)
evaluation.append(clf_log_cEXT.score(Test_X, Test_YEXT))
print("cEXT score: ", clf_log_cEXT.score(Test_X, Test_YEXT))

try:
    print("training Neuroticism cNEU using Logistic Regression...")
    clf_log_cNEU = LogisticRegression(solver="newton-cg")
    clf_log_cNEU.fit(Train_X, Train_YNEU)
    evaluation.append(clf_log_cNEU.score(Test_X, Test_YNEU))
    print("cNEU score: ", clf_log_cNEU.score(Test_X, Test_YNEU))
except:
    print("with this data not available (MBTI only 4 dimensions)")
    
print("training Agreeableness cAGR using using Logistic Regression...")
clf_log_cAGR = LogisticRegression(solver="newton-cg")
clf_log_cAGR.fit(Train_X, Train_YAGR)
evaluation.append(clf_log_cAGR.score(Test_X, Test_YAGR))
print("cAGR score: ", clf_log_cAGR.score(Test_X, Test_YAGR))

print("training Conscientiousness cCON using Logistic Regression...")
clf_log_cCON = LogisticRegression(solver="newton-cg")
clf_log_cCON.fit(Train_X, Train_YCON)
evaluation.append(clf_log_cCON.score(Test_X, Test_YCON))
print("cCON score: ", clf_log_cCON.score(Test_X, Test_YCON))

print("training Openness to Experience cOPN using Logistic Regression...")
clf_log_cOPN = LogisticRegression(solver="newton-cg")
clf_log_cOPN.fit(Train_X, Train_YOPN)
evaluation.append(clf_log_cOPN.score(Test_X, Test_YOPN))
print("cOPN score: ", clf_log_cOPN.score(Test_X, Test_YOPN))


# Random Forest

# In[15]:


from sklearn.ensemble import RandomForestClassifier
name="RF"


print("training Extraversion cEXT using Random Forest...")
clf_rf_cEXT = RandomForestClassifier(n_estimators=100)
clf_rf_cEXT.fit(Train_X, Train_YEXT)
evaluation.append(clf_rf_cEXT.score(Test_X, Test_YEXT))
print("cEXT score for RF: ", clf_rf_cEXT.score(Test_X, Test_YEXT))

try:
    print("training Neuroticism cNEU using Random Forest...")
    clf_rf_cNEU = RandomForestClassifier(n_estimators=100)
    clf_rf_cNEU.fit(Train_X, Train_YNEU)
    evaluation.append(clf_rf_cNEU.score(Test_X, Test_YNEU))
    print("cNEU score: ", clf_rf_cNEU.score(Test_X, Test_YNEU))
except:
    print("with this data not available (MBTI only 4 dimensions)")

print("training Agreeableness cAGR using using Random Forest...")
clf_rf_cAGR = RandomForestClassifier(n_estimators=100)
clf_rf_cAGR.fit(Train_X, Train_YAGR)
evaluation.append(clf_rf_cAGR.score(Test_X, Test_YAGR))
print("cAGR score: ", clf_rf_cAGR.score(Test_X, Test_YAGR))

print("training Conscientiousness cCON using Random Forest...")
clf_rf_cCON = RandomForestClassifier(n_estimators=100)
clf_rf_cCON.fit(Train_X, Train_YCON)
evaluation.append(clf_rf_cCON.score(Test_X, Test_YCON))
print("cCON score: ", clf_rf_cCON.score(Test_X, Test_YCON))

print("training Openness to Experience cOPN using Random Forest...")
clf_rf_cOPN = RandomForestClassifier(n_estimators=100)
clf_rf_cOPN.fit(Train_X, Train_YOPN)
evaluation.append(clf_rf_cOPN.score(Test_X, Test_YOPN))
print("cOPN score: ", clf_rf_cOPN.score(Test_X, Test_YOPN))


# In[16]:


for i in range(len(evaluation)):
    print(evaluation[i])

