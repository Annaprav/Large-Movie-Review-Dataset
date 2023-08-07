#!/usr/bin/env python
# coding: utf-8

# In[222]:


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.naive_bayes import GaussianNB
from nltk.stem import WordNetLemmatizer 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


# In[218]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
values_to_remove = ['no', 'nor', 'not',"don't", "should've",  'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stop = [x for x in stop if x not in values_to_remove]
print(stop)


# In[219]:


#функция для удаления стоп слов и лемматизации
def replace_stop(string):
    string = str.lower(string)
    mas = string.split()
    for i in stop:
        while i in mas :
            mas.remove(i)
    mas = list(map(lemmatizer.lemmatize,mas))
    string  = ' '.join(mas)
    return string

#функция для очистки отзыва от знаков припенания и приведение к нижнему регистру
def clean_text(data):
    patt = re.compile("[^\w\s]")
    
    data.loc[:, "text"] = data["text"].str.replace(patt, " ", regex=True)
    data['text'] = data['text'].apply(replace_stop)
    
    return data


# # Загрузка данных train 

# Загружаем данные из папки train в таблицу(столбцы - отзыв, рейтинг, оценка отзыва(neg/pos))

# In[220]:


train  = pd.DataFrame(columns = ['text', 'rating', 'label'])
#путь к папке train
data_path = "/Users/apple/Downloads/aclImdb/train"
pos = os.listdir(data_path + '/pos')
neg = os.listdir(data_path + '/neg')

for file in pos:
    with open(data_path + '/pos/'+ file) as f:
        contents = f.read()
    
    train.loc[len(train.index )] = [contents, int(file.split('_')[-1].split('.')[0]), 1]

for file in neg:
    with open(data_path + '/neg/'+ file) as f:
        contents = f.read()
    
    train.loc[len(train.index )] = [contents, int(file.split('_')[-1].split('.')[0]), 0]
   
    
clean_text(train)


# # Загрузка test

# In[221]:


test  = pd.DataFrame(columns = ['text', 'rating', 'label'])
#путь к папке test
data_path = "/Users/apple/Downloads/aclImdb/test"
pos = os.listdir(data_path + '/pos')
neg = os.listdir(data_path + '/neg')

for file in pos:
    with open(data_path + '/pos/'+ file) as f:
        contents = f.read()
    
    test.loc[len(test.index )] = [contents, int(file.split('_')[-1].split('.')[0]), 1]

for file in neg:
    with open(data_path + '/neg/'+ file) as f:
        contents = f.read()
    
    test.loc[len(test.index )] = [contents, int(file.split('_')[-1].split('.')[0]), 0]
   
    
clean_text(test)


# # Label

# In[224]:


X_train, y_train = train["text"],train["label"]
X_test, y_test = test['text'], test['label']
# Векторизация текста
vectorizer = TfidfVectorizer(max_features=5000)  
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()


# In[225]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test,y_pred,target_names=['Positive','Negative'])

print("Показатели тестовой модели")
print("Точность предсказания:", test_accuracy)
print(report)


# # Rating

# In[226]:


test['pred'] = y_pred
test


# In[256]:


X_train, y_train = train["text"],train["rating"]
X_test, y_test = test['text'], test['rating']

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()


# In[257]:


X_train = np.c_[X_train, train['label']]
X_test = np.c_[X_test, test['pred']]


# In[258]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

print("Показатели тестовой модели")
print("Точность предсказания:", test_accuracy)

