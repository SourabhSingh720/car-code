# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:25:09 2022

@author: forev
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv('car_data.csv')
#print(df.isnull().any())
#print(df.describe().unstack())

#df.boxplot(['AnnualSalary'])
df1=df.drop('User ID',axis=1)
#print(df1.head)

Male = pd.get_dummies(df1['Gender'],drop_first=True)
df2 = pd.concat([df1,Male], axis = 1)
#print(df2.head())

#no need of gender column
df3=df2.drop('Gender',axis=1)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df3.drop('Purchased',axis=1))
scaled_features = scaler.transform(df3.drop('Purchased',axis=1))


X= df3.drop('Purchased',axis = 1)
y = df3['Purchased']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.30)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators= 4)
rfc.fit(X_train,y_train)
y_pred =rfc.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))







