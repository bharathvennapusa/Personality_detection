#!/usr/bin/env python
# coding: utf-8

# # Model based on Bags of Words feature extraction

# In[1]:


import essay
import pickle


# In[2]:


# load the preprocessed data which we saved
# choose how much data you want to load (2467, 11142 or 89364)

#essays = pickle.load(open( "data/essays/essays2467.p", "rb"))

essays = pickle.load(open( "essays/essays11142.p", "rb"))

#essays = pickle.load(open( "data/essays/essays89364.p", "rb"))

print("loaded count of essays:", len(essays))


# # Split data in train & test

# In[3]:


from sklearn.model_selection import train_test_split
training, test = train_test_split(essays, test_size=0.20, random_state=42)


# In[4]:


train_x = [x.clean_text for x in training]

train_y_cEXT = [x.cEXT for x in training]
train_y_cNEU = [x.cNEU for x in training]
train_y_cAGR = [x.cAGR for x in training]
train_y_cCON = [x.cCON for x in training]
train_y_cOPN = [x.cOPN for x in training]


test_x = [x.clean_text for x in test]

test_y_cEXT = [x.cEXT for x in test]
test_y_cNEU = [x.cNEU for x in test]
test_y_cAGR = [x.cAGR for x in test]
test_y_cCON = [x.cCON for x in test]
test_y_cOPN = [x.cOPN for x in test]


# # bags of words

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer()

# create vectors from our words
train_x_vectors = bow_vectorizer.fit_transform(train_x)
test_x_vectors = bow_vectorizer.transform(test_x)
# # now that's a big thing :-O


# In[6]:


# for evaluation save some data for later:
evaluation = []
data = len(essays)
vec_name = "BoW"


# # SVM

# In[7]:


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

# In[8]:


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

# In[9]:


from sklearn.naive_bayes import GaussianNB
name = "gNB"
# clf_gnb = GaussianNB()
# clf_gnb.fit(train_x_vectors.toarray(), train_y)


print("training Extraversion cEXT using GaussianNaiveBayes...")
clf_gnb_cEXT = GaussianNB()
clf_gnb_cEXT.fit(train_x_vectors.toarray(), train_y_cEXT)
evaluation.append([data, vec_name, name, "cEXT", clf_gnb_cEXT.score(test_x_vectors.toarray(), test_y_cEXT)])
print("cEXT score: ", clf_gnb_cEXT.score(test_x_vectors.toarray(), test_y_cEXT))

try:
    print("training Neuroticism cNEU using GaussianNaiveBayes...")
    clf_gnb_cNEU = GaussianNB()
    clf_gnb_cNEU.fit(train_x_vectors.toarray(), train_y_cNEU)
    evaluation.append([data, vec_name, name, "cNEU", clf_gnb_cNEU.score(test_x_vectors.toarray(), test_y_cNEU)])
    print("cNEU score: ", clf_gnb_cNEU.score(test_x_vectors.toarray(), test_y_cNEU))
except:
    print("with this data not available (MBTI only 4 dimensions)")

    
print("training Agreeableness cAGR using using GaussianNaiveBayes...")
clf_gnb_cAGR = GaussianNB()
clf_gnb_cAGR.fit(train_x_vectors.toarray(), train_y_cAGR)
evaluation.append([data, vec_name, name, "cAGR", clf_gnb_cAGR.score(test_x_vectors.toarray(), test_y_cAGR)])
print("cAGR score: ", clf_gnb_cAGR.score(test_x_vectors.toarray(), test_y_cAGR))

print("training Conscientiousness cCON using GaussianNaiveBayes...")
clf_gnb_cCON = GaussianNB()
clf_gnb_cCON.fit(train_x_vectors.toarray(), train_y_cCON)
evaluation.append([data, vec_name, name, "cCON", clf_gnb_cCON.score(test_x_vectors.toarray(), test_y_cCON)])
print("cCON score: ", clf_gnb_cCON.score(test_x_vectors.toarray(), test_y_cCON))

print("training Openness to Experience cOPN using GaussianNaiveBayes...")
clf_gnb_cOPN = GaussianNB()
clf_gnb_cOPN.fit(train_x_vectors.toarray(), train_y_cOPN)
evaluation.append([data, vec_name, name, "cOPN", clf_gnb_cOPN.score(test_x_vectors.toarray(), test_y_cOPN)])
print("cOPN score: ", clf_gnb_cOPN.score(test_x_vectors.toarray(), test_y_cOPN))


# # Logisic Regression

# In[10]:


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


# In[11]:


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


# In[12]:


filename = "data/evaluation/evaluation" + str(data) + vec_name + ".p"
pickle.dump(evaluation, open(filename, "wb"))
print("evaluation saved as", filename)


# In[13]:


for i in range(len(evaluation)):
    print(evaluation[i][4])


# In[ ]:




