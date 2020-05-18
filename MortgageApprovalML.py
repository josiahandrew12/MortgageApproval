import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import warnings
import collections
import warnings
from collections import Counter
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense

DataFrame = pd.read_csv('MortgageApproval.csv')
DataFrame = DataFrame.dropna()
DataFrame.isna().any()
DataFrame=DataFrame.drop('Mortgage_ID', axis=1)
DataFrame['MortgageAmount']=(DataFrame['MortgageAmount']*100).astype(int)
Counter(DataFrame['Mortgage'])

pre_y=DataFrame['HomeLoan_Status']
pre_X=DataFrame.drop('Mortgage', axis=1)
dm_X=pd.get_dummies(pre_X)
dm_y=pre_y.map(dict(Y=1,N=0))
dm_X

smote = SMOTE(sampling_strategy='minority')
X1, y = smote.fit_sample(dm_X, dm_y)
sc = MinMaxScaler()
X = sc.fit_transform(X1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=True)

classifier = Sequential()
classifier.add(Dense(200, activation='relu', kernel_initializer='random_normal', input_dim=X_test.shape[1]))
classifier.add(Dense(400, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics =['accuracy'])
classifier.fit(X_train, y_train, batch_size=20, epochs=50, verbose=0)
eval_model = classifier.evaluate(X_train,y_train)

y_prediction = classifier.predict(X_train)
y_prediction = (y_prediction > 0.5)#any predicted value over 1 treat as approved

