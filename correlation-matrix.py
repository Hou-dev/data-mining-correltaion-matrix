#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import sklearn as skl
import sklearn.preprocessing as preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics

data = pd.read_csv('adult.data', delimiter = ',')
data.head()


# In[33]:


print(data.isnull().any())


# In[34]:


df_workclass=pd.get_dummies(data['Workclass'])
df_sector=pd.get_dummies(data['Sector'])
df_education=pd.get_dummies(data['Education'])
df_occupation=pd.get_dummies(data['Occupation'])
df_relationship = pd.get_dummies(data['Relationship'])
df_race = pd.get_dummies(data['Sex'])
df_country = pd.get_dummies(data['Native-Country'])


# In[43]:


df_final = pd.concat([data[['Age','Capital-Gain','Capital-Loss']],df_workclass,df_education,df_relationship,df_country],axis=1)


# In[44]:


data['Sex'].head()


# In[45]:


def get_y(y):
    if y.find("<=")>-1:
        return 0
    else:
        return 1


# In[46]:


# x = data[['Age','Sector','Education-num','Capital-Gain','Capital-Loss','Hours-Per-Week','Marital-Status']]
x= df_final
y = data['y'].apply(lambda y: get_y(y))


# In[47]:


def generate_auc(x,y):
    random_state = np.random.RandomState(0)
    x,y = shuffle(x,y,random_state=random_state)
    n_samples, n_features = x.shape
    half = int(n_samples/1.2)
    x_train, x_test = x[:half], x[half:]
    y_train, y_test = y[:half], y[half:]
    
    classifier = linear_model.LogisticRegression()
    probas_ = linear_model.LogisticRegression().fit(x_train,y_train).predict_proba(x_test)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:,1])
    roc_auc = metrics.auc(fpr,tpr)
    print( "Area under the ROC curve: %f" % roc_auc)
    return fpr,tpr, roc_auc, thresholds


# In[48]:


import random
from sklearn.utils import shuffle

generate_auc(x,y)


# In[25]:


# Encode the categorical features as numbers
# This code was used to represent data into correlation matrix
# taken from https://www.valentinmihov.com/2015/04/17/adult-income-data-set/
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

# Calculate the correlation and plot it
encoded_data, _ = number_encode_features(data)
sns.heatmap(encoded_data.corr(), square=True)
plt.show()


# In[ ]:




