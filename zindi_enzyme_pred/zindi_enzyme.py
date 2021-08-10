import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler

number_train = 100000


train_data = pd.read_csv("Train (1).csv")
train_data.drop(columns=['SEQUENCE_ID'], inplace=True)
train_data = train_data.iloc[:number_train,:]


def letter_to_number(letters, number_list):
	for i in letters:
		number_list.append(ord(i) - 96)

converted_to_number = []

for n in range(0, len(train_data['SEQUENCE'])):
	token_list = []
	token_list.append([i.lower() for i in train_data.iloc[n,0]])
	token_list = token_list[0]

	number_list = []
	letter_to_number(token_list,number_list)
	converted_to_number.append([number_list])

#import encdoded_val
#encoded_value = encdoded_val.converted_value()

encoded_df = pd.DataFrame(converted_to_number)

train_data.drop(columns=['SEQUENCE', 'CREATURE'], inplace=True)
train_data['encoded_value'] = encoded_df
listt = train_data['encoded_value'].apply(pd.Series) 

train_data = pd.merge(train_data,listt,on=train_data.index)
train_data.drop(columns=['key_0', 'encoded_value'], inplace=True)
train_data.fillna(value=0, inplace=True)

encoded_data = pd.get_dummies(train_data['LABEL'])
train_data = pd.merge(train_data, encoded_data, on=train_data.index)
train_data.drop(columns=['key_0', 'LABEL'], inplace=True)

filler = [0 for x in range(0,number_train)]
for i in range(1,6):
	train_data[1228+i] = filler

#print(train_data)

scaler = StandardScaler()
scaled = pd.DataFrame(scaler.fit_transform(train_data[range(0,1234)]))
train_data = pd.merge(encoded_data,scaled,on=train_data.index)
train_data.drop(columns=['key_0'], inplace=True)

#print(train_data)


#print(train_data.head())

import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(1234 , input_dim=1234, kernel_initializer='uniform', activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(20, kernel_initializer='uniform', activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

#print(encoded_data.shape)
X_train = train_data[range(0,1234)]
y_train = encoded_data

model.fit(X_train,y_train,epochs=30,batch_size=15)

model.save('model_weights.h5')


encoded_data.to_csv('encoded_data.csv')