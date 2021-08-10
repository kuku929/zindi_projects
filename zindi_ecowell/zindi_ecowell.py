#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score


# In[84]:


cd C:\Users\Krutarth\Desktop\Datasets\zindi_ecowell


# In[85]:


data = pd.read_csv('Train.csv')
data.head()


# In[86]:


#there are no common countries between train and test, thus we can drop it


# In[87]:


data.drop(columns=['ID', 'country'], inplace=True)


# In[88]:


data.head()


# In[89]:


data.isnull().sum()


# In[90]:


data.describe()


# In[91]:


#there seem to be outliers in the data.
#ghsl_pop_density,ghsl_pop_density,landcover_water_seasonal_10km_fraction


# In[92]:


data.drop(index=data.nighttime_lights.idxmax(), inplace=True)


# In[93]:


y_axis = [1]*len(data)
plt.scatter(data.nighttime_lights, data.Target)


# In[94]:


#might have to use polynomial features


# In[95]:


for column in data.columns:
    plt.scatter(data[column], y_axis)
    plt.title(column)
    plt.show()


# In[96]:


data.drop(data.nighttime_lights.idxmax(), inplace=True)


# In[97]:


data.drop(data[data.ghsl_built_1990_to_2000 >=0.43].index, inplace=True)


# In[98]:


data.drop(data[data.ghsl_built_2000_to_2014 >=0.42].index, inplace=True)


# In[99]:


data.drop(data[data.landcover_water_seasonal_10km_fraction >=38].index, inplace=True)


# In[100]:


data.drop(index=data[data.dist_to_shoreline >=1625].index, inplace=True)


# In[101]:


data.urban_or_rural.replace({
    'U':0,
    'R':1
}, inplace=True)


# In[102]:


len(data.year)


# In[103]:


scaler = MinMaxScaler()
X = data.drop(columns=['Target'])
y = data.Target
col = X.columns
df = pd.DataFrame(scaler.fit_transform(X))
df.columns = col
X = df


# In[104]:


X['merged_land'] = X.ghsl_not_built_up * X.landcover_urban_fraction
X.drop(columns=['ghsl_not_built_up', 'landcover_urban_fraction'], inplace=True)


# In[105]:


X['merged_light'] = X.nighttime_lights+X.ghsl_pop_density 


# In[106]:


X.drop(columns=['nighttime_lights', 'ghsl_pop_density','ghsl_built_pre_1975'], inplace=True)


# In[107]:


data_merged = pd.merge(X, y, on=X.index)
data_merged.drop(columns=['key_0'], inplace=True)


# In[108]:


plt.figure(figsize=(10,10))
sns.heatmap(data_merged.drop(columns=['year','urban_or_rural']).corr(), annot=True)


# In[109]:


X.isnull().sum()


# In[110]:


polynomial_fit = PolynomialFeatures(3)
data_pol = pd.DataFrame(polynomial_fit.fit_transform(X.drop(columns=['urban_or_rural'])))                  


# In[111]:


data_pol['urban_or_rural'] = X['urban_or_rural']


# In[112]:


hyper_parameter = {
    'n_estimators':[150,200],
    'max_depth':[1,2],
    'reg_alpha':[0.01, 0.2,1]
}


# In[113]:


regressor = XGBRegressor(random_state=0)


# In[114]:


grid_clf = GridSearchCV(regressor, hyper_parameter)


# In[115]:


X_train, X_test, y_train, y_test = train_test_split(data_pol, y, test_size=0.3, random_state=0)


# In[117]:


grid_clf.fit(X_train, y_train)


# In[118]:


y_pred= grid_clf.predict(X_test)


# In[119]:


score = r2_score(y_test, y_pred)
score


# In[ ]:


test.head()


# In[ ]:


test = pd.read_csv('Test.csv')


# In[ ]:


Id = pd.DataFrame(test['ID'])
test.drop(columns=['ID', 'country'], inplace=True)
test.urban_or_rural.replace({
    'U':0,
    'R':1
}, inplace=True)
test_df = scaler.transform(test)
test['merged_land'] = test.ghsl_not_built_up * test.landcover_urban_fraction

test.drop(columns=['ghsl_not_built_up', 'landcover_urban_fraction'], inplace=True)

test['merged_light'] = test.nighttime_lights+test.ghsl_pop_density

test.drop(columns=['nighttime_lights', 'ghsl_pop_density','ghsl_built_pre_1975'], inplace=True)


# In[ ]:


predictions = grid_clf.predict(test)
Id['Target'] = predictions
Id.to_csv('submission.csv', index=False)


# In[ ]:




