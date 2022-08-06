import os
import pandas as pd
import re
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

i = 0

def remove_words(text, word_list):
    global i
    if i % 100 == 0:
        print(i)
    x = text
    x = word_tokenize(x)
    x = [w for w in x if not w in word_list]
    result = ' '.join(x)
    i += 1
    return result


df = pd.read_csv('df_kaggle_cleaned.csv',encoding='utf-8', low_memory = False)## Read your own file
df_2 = pd.read_csv('words_to_remove.csv',encoding='utf-8', low_memory = False)## Read the file which has the words that are to be removed
df_2 = df_2.Words
word_list = df_2.values.tolist()
print ('Preapred word list')
#print(word_list[0])
#print (len(word_list))

df['TEXT_clean'] = df['TEXT'].apply(lambda x: remove_words(str(x), word_list))

df.to_csv(r'df_kaggle_cleaned_processed.csv', mode ='a', index = False) # final cleaned output file
