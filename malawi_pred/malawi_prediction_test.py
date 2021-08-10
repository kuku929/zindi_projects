#!/usr/bin/env python
# coding: utf-8

# In[11]:


import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json
import numpy as np
import pandas as pd
seed = 7


# In[12]:


cd C:\Users\Krutarth\Desktop\Datasets


# In[13]:


with open('malawi_pred.json', "r") as json_file:
    json_saved = json_file.read()
model=model_from_json(json_saved)    
model.summary()


# In[14]:


cd C:\Users\Krutarth\Desktop\Datasets


# In[15]:


data = pd.read_csv("Train.csv")
df=data[("Square_ID")]


# In[16]:


columns_2019 = []
columns_2015 = []

for col in data.columns:
    if '2019' in col:
        columns_2019.append(col)
    elif '2014' in col:
        columns_2015.append(col)
    elif '2015' in col:
        columns_2015.append(col)      


# In[17]:


test = data.drop(columns_2015, axis=1)
test=test.drop(['Square_ID'],axis=1)
test


# In[18]:


pred=model.predict(test)


# In[19]:


prede=pd.DataFrame(pred)
prede['Square_ID']=df
prede.head()
prede.describe()


# In[20]:


df=pd.read_csv('predictions3.csv')
df=df.rename(columns={"0":"target_2019"})
df=df[["Square_ID","target_2019"]]
df.head()
df.to_csv("predictions8.csv", index=False)


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




