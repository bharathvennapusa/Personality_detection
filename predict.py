#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pickle
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import pandas as pd
import re


# In[7]:

cEXT = pickle.load( open( "data/models/cEXT.p", "rb"))
cNEU = pickle.load( open( "data/models/cNEU.p", "rb"))
cAGR = pickle.load( open( "data/models/cAGR.p", "rb"))
cCON = pickle.load( open( "data/models/cCON.p", "rb"))
cOPN = pickle.load( open( "data/models/cOPN.p", "rb"))
vectorizer_31 = pickle.load( open( "data/models/vectorizer_31.p", "rb"))
vectorizer_30 = pickle.load( open( "data/models/vectorizer_30.p", "rb"))


# In[8]:


def predict_personality(text):
    scentences = re.split("(?<=[.!?]) +", text)
    text_vector_31 = vectorizer_31.transform(scentences)
    text_vector_30 = vectorizer_30.transform(scentences)
    EXT = cEXT.predict(text_vector_31)
    NEU = cNEU.predict(text_vector_30)
    AGR = cAGR.predict(text_vector_31)
    CON = cCON.predict(text_vector_31)
    OPN = cOPN.predict(text_vector_31)
    return [EXT[0], NEU[0], AGR[0], CON[0], OPN[0]]


# In[9]:


text = 'It is important to note that each of the five personality factors represents a range between two extremes. For example, extraversion represents a continuum between extreme extraversion and extreme introversion. In the real world, most people lie somewhere in between the two polar ends of each dimension. These five categories are usually described as follows.'


# In[10]:


predictions = predict_personality(text)
print("predicted personality:", predictions)
df = pd.DataFrame(dict(r=predictions, theta=['EXT','NEU','AGR', 'CON', 'OPN']))
fig = px.line_polar(df, r='r', theta='theta', line_close=True)
fig.show()


# In[ ]:




