#coding:utf-8
import numpy as np
from numpy import *
import pandas as pd
from sklearn import preprocessing


print 'start'

#print (85241/1.0)/(714*1.0)
print (61253/1.0)/(714*1.0)

F1 = 6.71755725/100.0; P = 0.05548550 ;R = F1*P/(2*P-F1)
print  '预测对了',528*R,'个'

F1 = 5.29363110/100.0; P = 0.04624277 ;R = F1*P/(2*P-F1)
print  '预测对了',528*R,'个'

p = 0.055; r = 44/528.0
print 'F1 =',2.0*p*r/(p+r)

'''
df = pd.read_csv('D:\\Documents\\SQL\\tianchi\\mobile_recommendation\\train.csv', header=None)
df = df.fillna(value=0)
#print df.head()
X = df.iloc[:,2:14].values
y = df.iloc[:,-1].values
'''

'''
a = array([[1,2],[3,4]])
b = [[1],[0]]
print sum(b,1)==0
print a[sum(b,1)==0]
'''


print 'finished'