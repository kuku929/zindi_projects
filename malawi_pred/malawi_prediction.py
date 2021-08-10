#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


# In[2]:


cd C:\Users\Krutarth\Desktop\Datasets


# In[3]:


data = pd.read_csv("Train.csv")


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


columns_2019 = []
columns_2015 = []

for col in data.columns:
    if '2019' in col:
        columns_2019.append(col)
    elif '2014' in col:
        columns_2015.append(col)
    elif '2015' in col:
        columns_2015.append(col)      


# In[7]:


train=data.drop(columns_2019, axis=1)
test=data.drop(columns_2015, axis=1)


# In[8]:


X = train.drop(['target_2015','Square_ID'], axis=1)
y = train[["target_2015"]]
X.describe()


# In[9]:


seed =7
def model():
    model = Sequential()
    model.add(Dense(21, input_dim=21, activation='relu', init="normal"))
    model.add(Dense(10, init='normal', activation='relu'))
    model.add(Dense(5, init='normal', activation='relu'))
    model.add(Dense(3,init='normal', activation='relu'))
    model.add(Dense(1, init="normal"))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model


# In[10]:


estimator = KerasRegressor(build_fn=model, epochs=50, batch_size=5, verbose=0)


# In[11]:


kfold = KFold(n_splits=10, random_state=seed)


# In[12]:


results = cross_val_score(estimator, X, y, cv=kfold)
print("standerdized result: %.10f" % (results.mean()))


# In[13]:


model=model()


# In[14]:


json_model=model.to_json()
with open('malawi_pred.json',"w") as json_file:
    json_file.write(json_model)                    


# In[15]:


model.save_weights('malawi_pred.h5')


# In[ ]:




