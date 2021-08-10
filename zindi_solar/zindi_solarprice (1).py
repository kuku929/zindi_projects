#!/usr/bin/env python
# coding: utf-8

# In[326]:


import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from statistics import mean
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor


# In[327]:


train = pd.read_csv(r'C:\Users\Krutarth\Desktop\Datasets\zindi_solarprice\Train.csv')
metadata = pd.read_csv(r'C:\Users\Krutarth\Desktop\Datasets\zindi_solarprice\metadata.csv')


# In[328]:


for i in range(0,len(train)):
    train.TransactionDates[i] = [x.strip().replace("'",'').replace('[','').replace(']','') for x in train.TransactionDates[i].split(',')] 

for i in range(0,len(train)):
    train.PaymentsHistory[i] = [float(x.strip().replace("'",'').replace('[','').replace(']','')) for x in train.PaymentsHistory[i].split(',')]


# In[329]:


train.columns


# In[330]:


#what i want to know:
#are all ids in train present in metadata? yes
#how do combine the 6 cols
#how do i convert transactiondates and paymenthistory to single value cols
#which cols in metadata are useful


# In[331]:


metadata.head()


# In[332]:


#idea:
#we will use m1...m6 to approximate a function and then use the coeffecients of these functions as the target


# In[333]:


month_pred = train[['m1', 'm2', 'm3', 'm4','m5', 'm6']]


# In[334]:


coeff_data = pd.DataFrame(columns=['c5','c4','c3','c2','c1','c0'],index=range(0,len(train)))
coefficients = []
for i in range(0,len(train)):
    coefficients.append(np.polyfit(range(1,7),month_pred.iloc[i,:],5))


# In[335]:


for i in range(0,len(train)):
    coeff_data.iloc[i,:] = coefficients[i]


# In[336]:


coeff_data


# In[337]:


metadata.drop(columns=['RegistrationDate'], inplace=True)


# In[338]:


metadata.head()


# In[339]:


#columns to take care of:
'''
UpsellDate           
PaymentMethod  
rateTypeEntity  
MainApplicantGender
Region -- will do after merge
Town 
Occupation -- will do after merge
SupplierName 
ExpectedTermDate  
FirstPaymentDate   
LastPaymentDate      
'''


# In[340]:


metadata.drop(columns=['UpsellDate'], inplace=True)


# In[341]:


metadata.drop(columns=['PaymentMethod'], inplace=True)


# In[342]:


metadata.drop(columns=['rateTypeEntity'], inplace=True)


# In[343]:


metadata.MainApplicantGender.replace({
    'Male':0,
    'Female':1
}, inplace=True)


# In[344]:


metadata.Region.value_counts()


# In[345]:


metadata.head()


# In[346]:


metadata.Occupation.value_counts()


# In[347]:


metadata.columns


# In[348]:


train['mean_payment'] = train[['m1', 'm2', 'm3', 'm4','m5', 'm6']].apply(lambda x: x.mean(), axis=1)


# In[349]:


metadata.drop(columns='SupplierName', inplace=True)


# In[350]:


metadata.ExpectedTermDate


# In[351]:


metadata['ExpectedTermDate'] = pd.to_datetime(metadata['ExpectedTermDate'])


# In[352]:


metadata['FirstPaymentDate'] = pd.to_datetime(metadata['FirstPaymentDate'])


# In[353]:


metadata['LastPaymentDate'] = pd.to_datetime(metadata['LastPaymentDate'])


# In[354]:


metadata['expected_time'] = metadata.ExpectedTermDate - metadata.FirstPaymentDate


# In[355]:


metadata['expected_time'] = metadata.expected_time.dt.days


# In[356]:


metadata['observed_time'] = metadata.LastPaymentDate - metadata.FirstPaymentDate


# In[357]:


metadata['observed_time'] = metadata.observed_time.dt.days


# In[358]:


metadata.drop(columns=['ExpectedTermDate','FirstPaymentDate', 'LastPaymentDate'], inplace=True)


# In[359]:


metadata.head()


# In[360]:


train.merge(metadata, on='ID',how='left')


# In[361]:


metadata.columns


# In[362]:


mean_data = metadata.dropna().groupby('MainApplicantGender')['Age'].mean()
mean_data


# In[363]:


metadata.Age.fillna(mean_data[0], inplace=True)


# In[364]:


for i in range(0,len(metadata)):
    if metadata.iloc[i,5] == 1 and metadata.iloc[i,6] == mean_data[0]:
        metadata.iloc[i, 6] = mean_data[1]


# In[365]:


metadata.isnull().sum()


# In[366]:


occ_region = metadata.groupby('Occupation')['Region'].apply(lambda x: x.value_counts().index[0]).reset_index()


# In[367]:


metadata = metadata.merge(occ_region, on='Occupation')


# In[368]:


metadata.drop(columns=['Region_x','Town'], inplace=True)


# In[369]:


metadata.head()


# In[370]:


metadata.info()


# In[371]:


train_merged = train.merge(metadata, on='ID',how='left')


# In[372]:


train_merged.isnull().sum()


# In[373]:


train_merged.info()


# In[374]:


train_merged.head()


# In[375]:


occupation_keys = train_merged.groupby('Occupation')['mean_payment'].mean().reset_index().sort_values(by='mean_payment')


# In[376]:


occupation_keys['occupation'] = range(0,len(occupation_keys))
occupation_keys.drop(columns=['mean_payment'], inplace=True)


# In[377]:


train_merged = train_merged.merge(occupation_keys, on='Occupation').drop(columns=['Occupation'])


# In[378]:


region_keys = train_merged.groupby('Region_y')['mean_payment'].mean().reset_index()


# In[379]:


region_keys


# In[380]:


region_keys['region'] = range(0,3)
region_keys.drop(columns=['mean_payment'], inplace=True)


# In[381]:


train_merged = train_merged.merge(region_keys, on='Region_y').drop(columns=['Region_y'])


# In[382]:


train_merged.shape


# In[383]:


time_duration = []
for i in range(len(train)):
    t = datetime.strptime(train['TransactionDates'][i][-1],'%m-%Y') - datetime.strptime(train['TransactionDates'][i][0],'%m-%Y')
    time_duration.append(t.days)


# In[384]:


train['duration'] = time_duration


# In[385]:


train.duration.idxmax()


# In[386]:


year_duration = []
month_duration = []


# In[387]:


transaction_col = []


# In[388]:


for y in range(len(train)):
    for i in range(len(train['TransactionDates'][y])):
        year_duration.append(datetime.strptime(train['TransactionDates'][y][i],'%m-%Y').year)
        month_duration.append(datetime.strptime(train['TransactionDates'][y][i],'%m-%Y').month)
    year_duration = [x-min(year_duration) for x in year_duration]
    time_columns=[str(year_duration[k])+str(month_duration[k]) for k in range(len(year_duration))]
    transaction_col.append(time_columns)
    year_duration = []
    month_duration = []


# In[389]:


time_col = []


# In[390]:


year = 0
month = 1
while year <=4:
    while month <=12:
        time_col.append(str(year)+str(month))
        month+=1
    year+=1
    month = 0        


# In[391]:


time_col


# In[392]:


#plan:
#we will first create a dictionary with the keys as the date and the items as the amount paid on that 
#date. Then create a dataset with the dates as the columns and the rows will have the payments on that
#date.lets scrap that, the resulting matrix is too sparse:(


# In[393]:


transaction_col[0]
train.PaymentsHistory[0]


# In[394]:


dict_0 = dict(zip(transaction_col[0], train.PaymentsHistory[0]))


# In[395]:


history_data = pd.DataFrame(columns=time_col,index=range(len(train)))


# In[396]:


for i in range(len(history_data)):
    dict_0 = dict(zip(transaction_col[i], train.PaymentsHistory[i]))
    history_data.iloc[i,:] = [dict_0.get(x) for x in history_data.columns]
    


# In[397]:


history_data.fillna(0, inplace=True)


# In[398]:


[x for x in history_data.columns if history_data[x].nunique()>100]


# In[399]:


for i in range(0,5):
    pay_coeff = np.polyfit(range(1,len(train.PaymentsHistory[i])+1),train.PaymentsHistory[i],4)
    poly = np.poly1d(pay_coeff)

    plt.plot(np.linspace(1,len(train.PaymentsHistory[i])), poly(np.linspace(1,len(train.PaymentsHistory[i]))))
    plt.plot(range(1,len(train.PaymentsHistory[i])+1),train.PaymentsHistory[i])
    plt.show()


# In[400]:


pay_coeff


# In[401]:


coeff_data1 = pd.DataFrame(columns=['p4','p3','p2','p1','p0'],index=range(0,len(train)))
coefficients1 = []
for i in train.PaymentsHistory:
    coefficients1.append(np.polyfit(range(1,len(i)+1),i,4))


# In[402]:


for i in range(0,len(train)):
    coeff_data1.iloc[i,:] = coefficients1[i]


# In[403]:


train_merged = train_merged.merge(coeff_data1, on=train.index)
train_merged['mean_payment'] = train_merged.PaymentsHistory.apply(lambda x: np.mean(x))
train_merged.drop(columns=['key_0', 'TransactionDates', 'PaymentsHistory'], inplace=True)


# In[404]:


train_merged.drop(columns=['m1','m2','m3','m4','m5','m6'], inplace=True)


# In[405]:


train_merged.drop(columns=['ID'], inplace=True)


# In[406]:


scaler = StandardScaler()
scaled_train = pd.DataFrame(scaler.fit_transform(train_merged))
scaled_train.columns = train_merged.columns


# In[407]:


scaled_train


# In[408]:


xgb = XGBRegressor(n_estimators=150, max_depth=2,n_jobs=10,random_state=0)
wrapper = RegressorChain(xgb,order=[0,1,2,3,4,5],random_state=0)


# In[409]:


wrapper_model = wrapper.fit(scaled_train, coeff_data)


# In[410]:


occupation_keys


# In[411]:


metadata.Region_y.value_counts()


# In[412]:


region_keys


# In[413]:


metadata_z = metadata.merge(occupation_keys, on='Occupation').drop(columns=['Occupation'])


# In[414]:


metadata_z = metadata_z.merge(region_keys, on='Region_y').drop(columns=['Region_y'])


# In[415]:


test = pd.read_csv(r'C:\Users\Krutarth\Desktop\Datasets\zindi_solarprice\Test.csv')


# In[416]:


test.head()


# In[417]:


test.PaymentsHistory[1]


# In[418]:


[float(x.strip().replace("'",'').replace('[','').replace(']','')) for x in test.PaymentsHistory[1].split(',')]


# In[419]:


for i in range(0,len(test)):
    test.TransactionDates[i] = [x.strip().replace("'",'').replace('[','').replace(']','') for x in test.TransactionDates[i].split(',')] 

for i in range(0,len(test)):
    test.PaymentsHistory[i] = [float(x.strip().replace("'",'').replace('[','').replace(']','')) for x in test.PaymentsHistory[i].split(',')]


# In[420]:


test.head()


# In[421]:


test.drop(columns=['TransactionDates'], inplace=True)


# In[422]:


test['mean_payment'] = test.PaymentsHistory.apply(lambda x: np.mean(x))


# In[423]:


test_merged = test.merge(metadata_z, on='ID')


# In[424]:


test_merged.head()


# In[425]:


coeff_dat2 = pd.DataFrame(columns=['q4','q3','q2','q1','q0'],index=range(0,len(test)))
coefficients2 = []
for i in test.PaymentsHistory:
    coefficients2.append(np.polyfit(range(1,len(i)+1),i,4))
for i in range(0,len(test)):
    coeff_dat2.iloc[i,:] = coefficients2[i]


# In[426]:


test_merged = test_merged.merge(coeff_dat2, on=test_merged.index)


# In[427]:


test_merged.drop(columns=['PaymentsHistory', 'key_0'], inplace=True)


# In[428]:


test_merged.drop(columns=['ID'], inplace=True)


# In[429]:


test_merged


# In[430]:


test_col = test_merged.columns
test_scaled = pd.DataFrame(scaler.transform(test_merged))
test_scaled.columns = test_col


# In[431]:


predicted_coeff = pd.DataFrame(wrapper_model.predict(test_scaled))


# In[432]:


future_val = pd.DataFrame(columns=['Target'],index=range(len(predicted_coeff)*6))
future_list = []
for x in range(len(predicted_coeff)):
    func = np.poly1d(np.array(predicted_coeff.iloc[x,:]))
    for i in range(1,7):
        future_list.append(func(i))        


# In[433]:


future_val['Target'] = future_list


# In[434]:


Id = test.ID


# In[435]:


Id = pd.DataFrame(Id)
Id = pd.concat([Id]*6).sort_index().reset_index(drop=True)


# In[436]:


for i in range(len(Id)):
    x+=1
    if x<7:
        Id.iloc[i] = Id.iloc[i]+' '+'x'+' '+'m'+str(x)
    else:
        x=1
        Id.iloc[i] = Id.iloc[i]+' '+'x'+' '+'m'+str(x)


# In[437]:


future_val['ID'] = Id


# In[438]:


cd C:\Users\Krutarth\Desktop\Datasets\zindi_solarprice


# In[439]:


future_val = future_val[['ID', 'Target']]
future_val.to_csv('submission.csv', index=False)


# In[440]:


future_val

