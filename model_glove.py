#!/usr/bin/env python
# coding: utf-8

# # Model based on GloVe feature extraction
# ## Global Vectors for Word Representation
# ### https://nlp.stanford.edu/projects/glove/
# GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

# In[1]:


import essay
import pickle
import numpy as np
import struct
import os
import pandas as pd

my_path = os.path.abspath(os.path.dirname('glove.6B/'))
GLOVE_BIG = os.path.join(my_path, "glove.6B.300d.txt")
GLOVE_SMALL = os.path.join(my_path, "glove.6B.50d.txt")
encoding="utf-8"


# In[3]:


# load the preprocessed data which we saved
# choose how much data you want to load (2467, 11142 or 89364)

essays = pickle.load(open( "essays/essays2467.p", "rb"))
#essays = pickle.load(open( "essays/essays11142.p", "rb"))
#essays = pickle.load(open( "essays/essays89364.p", "rb"))

print("loaded count of essays:", len(essays))


# # Vectorizer für Glove

# In[4]:


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(glove_mywords))])
        else:
            self.dim=0
            
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# # preparing the vectors - kinda manually... :S

# In[5]:


# load all vectors from all words from the GloVe File downladed from stanford
df = pd.read_csv(GLOVE_SMALL, sep=" ", quoting=3, header=None)
print(len(df))

# In[7]:


#load all words from all essays in a list
corpus = []
for e in essays:
    for w in e.words:
        corpus.append(w)
# and put it in a dataframe from this 
df_corpus = pd.DataFrame(corpus)
print(len(df_corpus))


# In[8]:


# inner join all GloVe Words with all words in the essays 
df_mywords = df.merge(df_corpus)
df_mywords = df_mywords.drop_duplicates()
df_mywords
print(len(df_mywords))

# In[9]:


#for the vectorizer we need a dict with all of "our" words
df_temp = df_mywords.set_index(0)
glove_mywords = {key: val.values for key, val in df_temp.T.items()}
glove_mywords
print(len(glove_mywords))

# In[10]:


# for every essay we save the GloVe Vectors in essay.glove as a dictionary
# 5min on 2400 essays and 300D

for e in essays:
    df_temp_e = pd.DataFrame(e.words)
    try:
        
        df_temp_e = df_temp_e.merge(df_mywords)
        df_temp_e = df_temp_e.drop_duplicates()
        df_temp_e = df_temp_e.set_index(0)    
        e.glove = {key: val.values for key, val in df_temp_e.T.items()}
    except:
        print("error")


# In[11]:


# save this essay data by converting into OBJECT essay and save with pickle and removing non emotional scentences
filename = "essays/essays_glove" + "50" + "d_" + str(len(essays)) + ".p"
pickle.dump(essays, open( filename, "wb"))
print("saved", len(essays), "entries: in", filename)


# # Split data in train & test

# In[67]:


from sklearn.model_selection import train_test_split
training, test = train_test_split(essays, test_size=0.20, random_state=42)


# In[68]:


train_x = [x.glove for x in training]

train_y_cEXT = [x.cEXT for x in training]
train_y_cNEU = [x.cNEU for x in training]
train_y_cAGR = [x.cAGR for x in training]
train_y_cCON = [x.cCON for x in training]
train_y_cOPN = [x.cOPN for x in training]


test_x = [x.glove for x in test]

test_y_cEXT = [x.cEXT for x in test]
test_y_cNEU = [x.cNEU for x in test]
test_y_cAGR = [x.cAGR for x in test]
test_y_cCON = [x.cCON for x in test]
test_y_cOPN = [x.cOPN for x in test]

train_x = np.array(train_x)
train_y_cEXT = np.array(train_y_cEXT)
train_y_cNEU = np.array(train_y_cNEU)
train_y_cAGR = np.array(train_y_cAGR)
train_y_cCON = np.array(train_y_cCON)
train_y_cOPN = np.array(train_y_cOPN)


# # Create Vectorizer for GloVe

# In[69]:


# the vectorizer calculates the MEAN of the vectors of all words 
# (that's what they recommend on stanford for a simple approach) 
glove_vectorizer = MeanEmbeddingVectorizer(glove_mywords)

# create mean from our vectors

train_x_vectors = glove_vectorizer.transform(train_x)

test_x_vectors = glove_vectorizer.transform(test_x)


# In[71]:


print(len(train_x_vectors))


# In[72]:


# for evaluation save some data for later:
evaluation = []
data = len(essays)
vec_name = "GloVe"


# # SVM

# In[73]:


from sklearn import svm
name = "svm"

print("training Extraversion cEXT using SVM...")
clf_svm_cEXT = svm.SVC(kernel='linear')
clf_svm_cEXT.fit(train_x_vectors, train_y_cEXT)
evaluation.append([data, vec_name, name, "cEXT", clf_svm_cEXT.score(test_x_vectors, test_y_cEXT)])
print("cEXT score: ", clf_svm_cEXT.score(test_x_vectors, test_y_cEXT))

try:
    print("training Neuroticism cNEU using SVM...")
    clf_svm_cNEU = svm.SVC(kernel='linear')
    clf_svm_cNEU.fit(train_x_vectors, train_y_cNEU)
    evaluation.append([data, vec_name, name, "cNEU", clf_svm_cNEU.score(test_x_vectors, test_y_cNEU)])
    print("cNEU score: ", clf_svm_cNEU.score(test_x_vectors, test_y_cNEU))
except:
    print("with this data not available (MBTI only 4 dimensions)")
    
print("training Agreeableness cAGR using using SVM...")
clf_svm_cAGR = svm.SVC(kernel='linear')
clf_svm_cAGR.fit(train_x_vectors, train_y_cAGR)
evaluation.append([data, vec_name, name, "cAGR", clf_svm_cAGR.score(test_x_vectors, test_y_cAGR)])

print("cAGR score: ", clf_svm_cAGR.score(test_x_vectors, test_y_cAGR))

print("training Conscientiousness cCON using SVM...")
clf_svm_cCON = svm.SVC(kernel='linear')
clf_svm_cCON.fit(train_x_vectors, train_y_cCON)
evaluation.append([data, vec_name, name, "cCON", clf_svm_cCON.score(test_x_vectors, test_y_cCON)])
print("cCON score: ", clf_svm_cCON.score(test_x_vectors, test_y_cCON))

print("training Openness to Experience cOPN using SVM...")
clf_svm_cOPN = svm.SVC(kernel='linear')
clf_svm_cOPN.fit(train_x_vectors, train_y_cOPN)
evaluation.append([data, vec_name, name, "cOPN", clf_svm_cOPN.score(test_x_vectors, test_y_cOPN)])
print("cOPN score: ", clf_svm_cOPN.score(test_x_vectors, test_y_cOPN))


# # Decision Tree

# In[74]:


from sklearn import tree
name = "tree"

print("training Extraversion cEXT using dec...")
clf_dec_cEXT = tree.DecisionTreeClassifier()
clf_dec_cEXT.fit(train_x_vectors, train_y_cEXT)
evaluation.append([data, vec_name, name, "cEXT", clf_dec_cEXT.score(test_x_vectors, test_y_cEXT)])

print("cEXT score: ", clf_dec_cEXT.score(test_x_vectors, test_y_cEXT))

try:
    print("training Neuroticism cNEU using dec...")
    clf_dec_cNEU = tree.DecisionTreeClassifier()
    clf_dec_cNEU.fit(train_x_vectors, train_y_cNEU)
    evaluation.append([data, vec_name, name, "cNEU", clf_dec_cNEU.score(test_x_vectors, test_y_cNEU)])
    print("cNEU score: ", clf_dec_cNEU.score(test_x_vectors, test_y_cNEU))
except:
    print("with this data not available (MBTI only 4 dimensions)")

print("training Agreeableness cAGR using using dec...")
clf_dec_cAGR = tree.DecisionTreeClassifier()
clf_dec_cAGR.fit(train_x_vectors, train_y_cAGR)
evaluation.append([data, vec_name, name, "cAGR", clf_dec_cAGR.score(test_x_vectors, test_y_cAGR)])
print("cAGR score: ", clf_dec_cAGR.score(test_x_vectors, test_y_cAGR))

print("training Conscientiousness cCON using dec...")
clf_dec_cCON = tree.DecisionTreeClassifier()
clf_dec_cCON.fit(train_x_vectors, train_y_cCON)
evaluation.append([data, vec_name, name, "cCON", clf_dec_cCON.score(test_x_vectors, test_y_cCON)])
print("cCON score: ", clf_dec_cCON.score(test_x_vectors, test_y_cCON))

print("training Openness to Experience cOPN using dec...")
clf_dec_cOPN = tree.DecisionTreeClassifier()
clf_dec_cOPN.fit(train_x_vectors, train_y_cOPN)
evaluation.append([data, vec_name, name, "cOPN", clf_dec_cOPN.score(test_x_vectors, test_y_cOPN)])
print("cOPN score: ", clf_dec_cOPN.score(test_x_vectors, test_y_cOPN))


# # Naive Bayes

# In[75]:


from sklearn.naive_bayes import GaussianNB
name = "gNB"
# clf_gnb = GaussianNB()
# clf_gnb.fit(train_x_vectors, train_y)


print("training Extraversion cEXT using GaussianNaiveBayes...")
clf_gnb_cEXT = GaussianNB()
clf_gnb_cEXT.fit(train_x_vectors, train_y_cEXT)
evaluation.append([data, vec_name, name, "cEXT", clf_gnb_cEXT.score(test_x_vectors, test_y_cEXT)])
print("cEXT score: ", clf_gnb_cEXT.score(test_x_vectors, test_y_cEXT))

try:
    print("training Neuroticism cNEU using GaussianNaiveBayes...")
    clf_gnb_cNEU = GaussianNB()
    clf_gnb_cNEU.fit(train_x_vectors, train_y_cNEU)
    evaluation.append([data, vec_name, name, "cNEU", clf_gnb_cNEU.score(test_x_vectors, test_y_cNEU)])
    print("cNEU score: ", clf_gnb_cNEU.score(test_x_vectors, test_y_cNEU))
except:
    print("with this data not available (MBTI only 4 dimensions)")

    
print("training Agreeableness cAGR using using GaussianNaiveBayes...")
clf_gnb_cAGR = GaussianNB()
clf_gnb_cAGR.fit(train_x_vectors, train_y_cAGR)
evaluation.append([data, vec_name, name, "cAGR", clf_gnb_cAGR.score(test_x_vectors, test_y_cAGR)])
print("cAGR score: ", clf_gnb_cAGR.score(test_x_vectors, test_y_cAGR))

print("training Conscientiousness cCON using GaussianNaiveBayes...")
clf_gnb_cCON = GaussianNB()
clf_gnb_cCON.fit(train_x_vectors, train_y_cCON)
evaluation.append([data, vec_name, name, "cCON", clf_gnb_cCON.score(test_x_vectors, test_y_cCON)])
print("cCON score: ", clf_gnb_cCON.score(test_x_vectors, test_y_cCON))

print("training Openness to Experience cOPN using GaussianNaiveBayes...")
clf_gnb_cOPN = GaussianNB()
clf_gnb_cOPN.fit(train_x_vectors, train_y_cOPN)
evaluation.append([data, vec_name, name, "cOPN", clf_gnb_cOPN.score(test_x_vectors, test_y_cOPN)])
print("cOPN score: ", clf_gnb_cOPN.score(test_x_vectors, test_y_cOPN))


# # Logisic Regression

# In[76]:


from sklearn.linear_model import LogisticRegression
name="logR"
print("training Extraversion cEXT using Logistic Regression...")
clf_log_cEXT = LogisticRegression(solver="newton-cg")
clf_log_cEXT.fit(train_x_vectors, train_y_cEXT)
evaluation.append([data, vec_name, name, "cEXT", clf_log_cEXT.score(test_x_vectors, test_y_cEXT)])
print("cEXT score: ", clf_log_cEXT.score(test_x_vectors, test_y_cEXT))

try:
    print("training Neuroticism cNEU using Logistic Regression...")
    clf_log_cNEU = LogisticRegression(solver="newton-cg")
    clf_log_cNEU.fit(train_x_vectors, train_y_cNEU)
    evaluation.append([data, vec_name, name, "cNEU", clf_log_cNEU.score(test_x_vectors, test_y_cNEU)])
    print("cNEU score: ", clf_log_cNEU.score(test_x_vectors, test_y_cNEU))
except:
    print("with this data not available (MBTI only 4 dimensions)")
    
print("training Agreeableness cAGR using using Logistic Regression...")
clf_log_cAGR = LogisticRegression(solver="newton-cg")
clf_log_cAGR.fit(train_x_vectors, train_y_cAGR)
evaluation.append([data, vec_name, name, "cAGR", clf_log_cAGR.score(test_x_vectors, test_y_cAGR)])
print("cAGR score: ", clf_log_cAGR.score(test_x_vectors, test_y_cAGR))

print("training Conscientiousness cCON using Logistic Regression...")
clf_log_cCON = LogisticRegression(solver="newton-cg")
clf_log_cCON.fit(train_x_vectors, train_y_cCON)
evaluation.append([data, vec_name, name, "cCON", clf_log_cCON.score(test_x_vectors, test_y_cCON)])
print("cCON score: ", clf_log_cCON.score(test_x_vectors, test_y_cCON))

print("training Openness to Experience cOPN using Logistic Regression...")
clf_log_cOPN = LogisticRegression(solver="newton-cg")
clf_log_cOPN.fit(train_x_vectors, train_y_cOPN)
evaluation.append([data, vec_name, name, "cOPN", clf_log_cOPN.score(test_x_vectors, test_y_cOPN)])
print("cOPN score: ", clf_log_cOPN.score(test_x_vectors, test_y_cOPN))


# # Random Forest

# In[77]:


from sklearn.ensemble import RandomForestClassifier
name="RF"


print("training Extraversion cEXT using Random Forest...")
clf_rf_cEXT = RandomForestClassifier(n_estimators=100)
clf_rf_cEXT.fit(train_x_vectors, train_y_cEXT)
evaluation.append([data, vec_name, name, "cEXT", clf_rf_cEXT.score(test_x_vectors, test_y_cEXT)])
print("cEXT score: ", clf_rf_cEXT.score(test_x_vectors, test_y_cEXT))

try:
    print("training Neuroticism cNEU using Random Forest...")
    clf_rf_cNEU = RandomForestClassifier(n_estimators=100)
    clf_rf_cNEU.fit(train_x_vectors, train_y_cNEU)
    evaluation.append([data, vec_name, name, "cNEU", clf_rf_cNEU.score(test_x_vectors, test_y_cNEU)])
    print("cNEU score: ", clf_rf_cNEU.score(test_x_vectors, test_y_cNEU))
except:
    print("with this data not available (MBTI only 4 dimensions)")

print("training Agreeableness cAGR using using Random Forest...")
clf_rf_cAGR = RandomForestClassifier(n_estimators=100)
clf_rf_cAGR.fit(train_x_vectors, train_y_cAGR)
evaluation.append([data, vec_name, name, "cAGR", clf_rf_cAGR.score(test_x_vectors, test_y_cAGR)])
print("cAGR score: ", clf_rf_cAGR.score(test_x_vectors, test_y_cAGR))

print("training Conscientiousness cCON using Random Forest...")
clf_rf_cCON = RandomForestClassifier(n_estimators=100)
clf_rf_cCON.fit(train_x_vectors, train_y_cCON)
evaluation.append([data, vec_name, name, "cCON", clf_rf_cCON.score(test_x_vectors, test_y_cCON)])
print("cCON score: ", clf_rf_cCON.score(test_x_vectors, test_y_cCON))

print("training Openness to Experience cOPN using Random Forest...")
clf_rf_cOPN = RandomForestClassifier(n_estimators=100)
clf_rf_cOPN.fit(train_x_vectors, train_y_cOPN)
evaluation.append([data, vec_name, name, "cOPN", clf_rf_cOPN.score(test_x_vectors, test_y_cOPN)])
print("cOPN score: ", clf_rf_cOPN.score(test_x_vectors, test_y_cOPN))


# In[79]:


filename = "data/evaluation/evaluation" + str(data) + vec_name + ".p"
pickle.dump(evaluation, open(filename, "wb"))
print("evaluation saved as", filename)


# In[78]:


print(evaluation)


# In[ ]:




