import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler


test = pd.read_csv("Test.csv")

def letter_to_number(letters, number_list):
	for i in letters:
		number_list.append(ord(i.lower()) - 96)

converted_to_number = []

import encdoded_val as enc 
converted_to_number = enc.conv_test()
#print(converted_to_number[0])
encoded_df = pd.DataFrame(converted_to_number)
test['encoded_value'] = encoded_df
listt = test['encoded_value'].apply(pd.Series) 
#test = pd.merge(test,listt,on=test.index)

#test.to_csv('test.csv')
#print(listt)

encoded_data = pd.read_csv('encoded_data.csv')

encoded_data.drop(columns=['Unnamed: 0'], inplace=True)

encoded_test = pd.read_csv('encoded.csv')
encoded_test.drop(columns=['Unnamed: 0'], inplace=True)
encoded_test.fillna(value=0,inplace=True)

scaler = StandardScaler()
encoded_test = scaler.fit_transform(encoded_test)

from keras.models import load_model

model = load_model('model_weights.h5')
predictions = pd.DataFrame(model.predict(encoded_test), columns=encoded_data.columns)
print(predictions)
predictions.to_csv('predictions.csv',index=False)


#print(predictions.idxmax(1))
predictions = pd.read_csv('predictions.csv')
#predictions.drop(columns='Unnamed: 0', inplace=True)
test['LABEL'] = predictions.idxmax(1)


test.drop(columns=['Unnamed: 0', 'Unnamed: 0.1','CREATURE','SEQUENCE','encoded_value'],inplace=True)
test.to_csv('predicted_result.csv', index=False)
'''
predicted_result = pd.read_csv('predicted_result.csv')
predicted_result.drop(columns=['encoded_value'], inplace=True)
predicted_result.to_csv('predicted_result.csv', index=False)
print(predicted_result.columns)
