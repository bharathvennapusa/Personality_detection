#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import pandas as pd
import re
import os


# In[2]:


cEXT = pickle.load( open( "data/models/clf_rf_cEXT.p", "rb"))
cAGR = pickle.load( open( "data/models/clf_rf_cAGR.p", "rb"))
cCON = pickle.load( open( "data/models/clf_rf_cCON.p", "rb"))
cOPN = pickle.load( open( "data/models/clf_rf_cOPN.p", "rb"))
vectorizer_31 = pickle.load( open( "data/models/vectorizer_31.p", "rb"))
vectorizer_30 = pickle.load( open( "data/models/vectorizer_30.p", "rb"))


# In[3]:


def predict_personality(dflist):
    EXT = cEXT.predict(dflist)
    AGR = cAGR.predict(dflist)
    CON = cCON.predict(dflist)
    OPN = cOPN.predict(dflist)
    d = [EXT, AGR, CON, OPN]
    return d


# In[4]:


df = pd.read_csv(r'to_export.merged.crm_english.liwcanalysis_social_handle.csv', encoding='utf-8', low_memory = False)
df


# In[6]:


test = df.iloc[:,4:]
test
output = predict_personality(test)


# In[7]:


df['oEXT'] = output[0]
df['oAGR'] = output[1]
df['oCON'] = output[2]
df['oOPN'] = output[3]
df


# In[ ]:


df.to_csv(r'predictions_for_social.csv', mode ='a', index = False) # final cleaned output file

