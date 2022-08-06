#!/usr/bin/env python
# coding: utf-8

# # Preprocessing of data
# 

# In[1]:


import pandas as pd
import essay
import numpy as np
import pickle
import re


# In[2]:


def mbti_to_big5(mbti):
    # check https://en.wikipedia.org/wiki/Myers%E2%80%93Briggs_Type_Indicator
    # in mbti (myers briggs) ther is invrovert vs. extrovert
    # which corellates with Extroversion in BIG FIVE
    mbti = mbti.lower()
    cEXT, cNEU, cAGR, cCON, cOPN = 0,np.NaN,0,0,0
    if mbti[0] == "i":
        cEXT = 0
    elif mbti[0] == "e":
        cEXT = 1

    # in mbti (myers briggs) ther is I*N*TUITION vs SENSING
    # which corellates with OPENNESS in BIG FIVE
    if mbti[1] == "n":
        cOPN = 1
    elif mbti[1] == "s":
        cOPN = 0   

    # in mbti (myers briggs) ther is THINKER vs FEELER
    # which corellates with AGREEABLENESS in BIG FIVE
    if mbti[2] == "t":
        cAGR = 0
    elif mbti[2] == "f":
        cAGR = 1

    # in mbti (myers briggs) ther is JUDGER vs PERCEIVER
    # which corellates with CONSCIENTIOUSNESS in BIG FIVE (worst corellation)
    # especially bec. orderlyness corellates with conscientiousness
    if mbti[3] == "p":
        cCON = 0
    elif mbti[3] == "j":
        cCON = 1

    return cEXT, cNEU, cAGR, cCON, cOPN


# In[3]:


def remove_unemotional_scentences(emotional_words, text_as_one_string):
    reduced_s = ""
    scentences = re.split('(?<=[.!?]) +', text_as_one_string)
    for s in scentences:
        if any(e in s for e in emotional_words):
            reduced_s = reduced_s + s + " "
        else:
            pass
    return reduced_s


# In[4]:


# simply put every row of our read dataframe into a list of 
# the object "Essay"
# remove data from list substract
def create_essays(df, subtract=None):
    essays = []
    for index, row in df.iterrows():
        essays.append(essay.Essay(row.TEXT, row.cEXT, row.cNEU, row.cAGR, row.cCON, row.cOPN))  

    # remove scentences which do not contain emotionally charged words 
    # from the emotional lexicon
    if subtract != None:
        for x in essays:
            x.filtered_text = remove_unemotional_scentences(emotional_words, x.clean_text)

    return essays


# # Loading the essays from paper
# ## scientific gold standard
# ## https://sentic.net/deep-learning-based-personality-detection.pdf
# ### (scientific gold standard "stream of counsciousness" essays labeled with personality traits of the big five)

# In[5]:


# we read in the data from "essays.csv" and 
# "essays.csv" contains all essays with classification 
df_essays = pd.read_csv('data/training/essays.csv', encoding='cp1252', delimiter=',', quotechar='"')

# for every essay, we replace the personalitiy categories 
# of the essay wich are "y" and "n" with "1" and "0" 
for e in df_essays.columns[2:7]:
    df_essays[e] = df_essays[e].replace('n', '0')
    df_essays[e] = df_essays[e].replace('y', '1')
    # not sure if we need this line: furter investigation possible:
    df_essays[e] = pd.to_numeric(df_essays[e])

df_essays = df_essays[["TEXT", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"]]
df_essays


# # Load the MBTI kaggle dataset
# ## https://www.kaggle.com/datasnaek/mbti-type

# In[7]:


df_kaggle = pd.read_csv('data/training/mbti_1.csv',  skiprows=0 )

df_kaggle["cEXT"] =   df_kaggle.apply(lambda x: mbti_to_big5(x.type)[0], 1)
df_kaggle["cNEU"] =   df_kaggle.apply(lambda x: mbti_to_big5(x.type)[1], 1)
df_kaggle["cAGR"] =   df_kaggle.apply(lambda x: mbti_to_big5(x.type)[2], 1)
df_kaggle["cCON"] =   df_kaggle.apply(lambda x: mbti_to_big5(x.type)[3], 1)
df_kaggle["cOPN"] =   df_kaggle.apply(lambda x: mbti_to_big5(x.type)[4], 1)

df_kaggle = df_kaggle[["posts", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"]]
df_kaggle.columns = ["TEXT", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"]

# remove som fancy ||| things
df_kaggle["TEXT"] = df_kaggle.apply(lambda x: x.TEXT.replace("|||", " ")[:], 1)


df_kaggle


## # Load Reddit Dataset
## ## Matej Gjurković  https://peopleswksh.github.io/pdf/PEOPLES12.pdf
## ### received on request
##
#
## In[8]:
#
#
## the file is kinda huge. thanks Matej
#file = 'data/training/typed_comments.csv'
#df_reddit = pd.read_csv(file)
#
#
## In[9]:
#
#
##remove some rows to to keep longer text (the 420 because it makes avg word count compareable to the rest of the data)
#df_reddit = df_reddit[df_reddit.word_count > 420]
#
#df_reddit["cEXT"] =   df_reddit.apply(lambda x: mbti_to_big5(x.type)[0], 1)
#df_reddit["cNEU"] =   df_reddit.apply(lambda x: mbti_to_big5(x.type)[1], 1)
#df_reddit["cAGR"] =   df_reddit.apply(lambda x: mbti_to_big5(x.type)[2], 1)
#df_reddit["cCON"] =   df_reddit.apply(lambda x: mbti_to_big5(x.type)[3], 1)
#df_reddit["cOPN"] =   df_reddit.apply(lambda x: mbti_to_big5(x.type)[4], 1)
#df_reddit = df_reddit[["comment", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"]]
#df_reddit.columns = ["TEXT", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
#df_reddit.reset_index(drop=True, inplace=True)
#df_reddit


# # Load Emotional Lexicon to substract from data

# In[10]:


# also from "Emotional_Lexicon.csv" we read in the data, which is a list of words and 
# has several categories of emotions. 
# anger - anticipation - disgust - fear - joy - negative - positive 
# - sadness - surprise - trust - Charged
df_lexicon = pd.read_csv('data/training/Emotion_Lexicon.csv', index_col=0)


# some of the words have no emotional category, 
# so let's remove them as they have no use to us.
# can be improved by not even loading them when all columns are 0. maybe later.
df_lexicon = df_lexicon[(df_lexicon.T != 0).any()]
emotional_words = df_lexicon.index.tolist()


# # Concatinate the datasets for 3 different data to work with and compare to

# ## Create Data Base 1 - only esseys.csv - and save as object list

# In[11]:


# save preprocessed data by converting into OBJECT essay and save with pickle and removing non emotional scentences
essays = create_essays(df_essays, emotional_words)
pickle.dump(essays, open( "essays/essays2467.p", "wb"))
print("saved entries: ", len(essays))


# ## Create Data Base 2 - Essay Data and Kaggle data - and save as object list

# In[12]:


# concatinate the dataframes:
frames  = [df_essays, df_kaggle]
essays_kaggle = pd.concat(frames, sort=False)
essays_kaggle.reset_index(drop=True)

# preprocess data by converting into OBJECT essay and save with pickle and removing non emotional scentences
essays_kaggle = create_essays(essays_kaggle, emotional_words)
pickle.dump(essays_kaggle, open("essays/essays11142.p", "wb"))
print("saved entries: ", len(essays_kaggle))


## ## Create Data Base 3 - Essay Data and Kaggle data and Reddit data - and save as object list
#
## In[13]:
#
#
## concatinate the dataframes:
#frames  = [df_essays, df_kaggle, df_reddit]
#essays_kaggle_reddit = pd.concat(frames, sort=False)
#essays_kaggle_reddit.reset_index(drop=True)
#
## preprocess data by converting into OBJECT essay and save with pickle and removing non emotional scentences
#essays_kaggle_reddit = create_essays(essays_kaggle_reddit, emotional_words)
#pickle.dump(essays_kaggle_reddit, open("essays/essays89364.p", "wb"))
#print("saved entries: ", len(essays_kaggle_reddit))
