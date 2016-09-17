#coding:utf-8
import numpy as np
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

print 'start'

df = pd.read_csv('D:\\Documents\\SQL\\tianchi\\mobile_recommendation\\train.csv')
df = df.fillna(value=0)

X = df.iloc[:,2:14].values
y = df['label'].values
X_y = df.iloc[:,2:15].values
X_y_N = array(X_y[X_y[:,-1]==0])
X_y_P = array(X_y[X_y[:,-1]==1])
i = 0
X_y_sample = []
while(i<shape(X_y_N)[0]):
    X_y_sample.append(X_y_N[i])
    i+=10
X_y_sample = array(X_y_sample)
X_y_sample = np.concatenate((X_y_sample,X_y_P,X_y_P))

plt.scatter(X_y_sample[:,2], X_y_sample[:,3], s=100, c=15.0*(X_y_sample[:,-1]+1))
plt.show()


