#coding:utf-8
import numpy as np
from numpy import *
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing

from sklearn.cluster import KMeans

print 'start'

df = pd.read_csv('D:\\Documents\\SQL\\tianchi\\mobile_recommendation\\train.csv', header=None) # header=None
df = df.fillna(value=0)

# 14 -1
n_col = -1

X = df.iloc[:,2:n_col].values
y = df.iloc[:,-1:].values
X_y = pd.concat([df.iloc[:,2:n_col], df.iloc[:,-1:]], axis=1)
X_y = X_y.values
print '原训练集大小:',shape(X_y)
X_y = X_y[sum(X_y[:,0:-1],1)>0.0,:]
print '过滤后大小:',shape(X_y)
X_y_N = array(X_y[X_y[:,-1]==0])
X_y_P = array(X_y[X_y[:,-1]==1])

#--输出正样本的csv文件
#pd.DataFrame(X_y_P).to_csv('D:\\Documents\\SQL\\tianchi\\mobile_recommendation\\X_y_P.csv' 
#                           , index=False, header=None, encoding='utf-8')

print '负样本数：', shape(X_y_N)
print '正样本数：', shape(X_y_P)
print '负正比为：', shape(X_y_N)[0]/float(shape(X_y_P)[0])

i = 0
N_rate = 13  #----负样本下采样率------------------------
X_y_sample = []
while(i<shape(X_y_N)[0]):
    X_y_sample.append(X_y_N[i])
    i+=N_rate
X_y_sample = array(X_y_sample)
X_y_sample = np.concatenate((X_y_sample,X_y_P,X_y_P))

#model = LogisticRegression()  # class_weight='balanced'
# criterion='entropy', max_features='auto', max_depth=3, min_samples_leaf=5, class_weight='balanced'
#model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)  
#model = GaussianNB()
#model = svm.SVC(C=10)
#model = KNeighborsClassifier(n_neighbors=5) 
#model = SGDClassifier()
#model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3))
model = RandomForestClassifier(class_weight='balanced')  # class_weight='balanced'
#model = GradientBoostingClassifier()
#model = KMeans(2)
#X_normalize = preprocessing.normalize(X_y_sample[:,0:-1], norm='l1', axis=0) 
model.fit(X_y_sample[:,0:-1], X_y_sample[:,-1])
#print(model)

# 训练集的自身拟合
print 'train_set:'
expected = y
#X_normalize = preprocessing.normalize(X, norm='l1', axis=0)
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

'''
# 测试集
for i in [1]:
    path_name = 'D:\\Documents\\SQL\\tianchi\\mobile_recommendation\\test'+str(i)+'.csv'
    df2 = pd.read_csv(path_name, header=None)
    df2 = df2.fillna(value=0)
    print '\ntest_set'+str(i)+':'
    X_test = df2.iloc[:,2:n_col].values
    y_test = df2.iloc[:,-1:].values
    expected = y_test
    #X_normalize = preprocessing.normalize(X_test, norm='l1', axis=0)
    predicted = model.predict(X_test)
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
'''


# predict
df = pd.read_csv('D:\\Documents\\SQL\\tianchi\\mobile_recommendation\\predict.csv')
df = df.fillna(value=0)
X = df.iloc[:,2:].values
predicted = model.predict(X)
df['predict'] = pd.Series(predicted)
df = df[df.predict==1]
df.iloc[:,0:2].to_csv('D:\\Documents\\SQL\\tianchi\\mobile_recommendation\\tianchi_mobile_recommendation_predict.csv', index=False, encoding='utf-8')



print 'finished'

