# -*- coding: utf-8 -*-
"""
Created on 25-TUE-2018

@author: Rocky Jain
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing,cross_validation,svm

df = pd.read_csv(r'C:\Users\Rocky Jain\MachineLearning Breast Cancer- Original(UCI DATA SET)\weight_dataset\balance-scale.data.txt')

encoder = preprocessing.LabelEncoder()
df['class'] = encoder.fit_transform(df['class'])

X = np.array(df.drop(['class'],1))
Y = np.array(df['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X,Y,test_size=0.2)


classifier = svm.SVC(kernel='poly')
classifier.fit(x_train, y_train)

accuracy = classifier.score(x_test,y_test)


/**
  predicting the label using random features
*/
predict = classifier.predict([[1,2,3,4],
                              [5,4,3,1],
                              [0,0,0,0],
                              [0,2,2,0],
                              [0,1,4,3]])
predict = encoder.inverse_transform(predict)
print("via SVC: ",accuracy,predict)
