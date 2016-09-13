#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import pandas as pd
import time
from def_fun import test_stationarity 
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARIMA,ARMA

print 'start>>'

df=pd.read_csv('D:\\Documents\\SQL\\tianchi\\profile_predict\\all_per_day.csv'
               , parse_dates=True, index_col='time')

df=df.dropna(how='any')
#print df.head()

'''
# 直接读取csv，不用数据库统计
user_balance = pd.read_csv('D:\\Documents\\SQL\\tianchi\\profile_predict\\user_balance_table.csv', parse_dates = ['report_date'])
timeGroup = user_balance.groupby(['report_date'])
purchaseRedeemTotal = timeGroup['total_purchase_amt', 'total_redeem_amt'].sum()
#print purchaseRedeemTotal.head()
'''

ts_p = df['purchase'].astype(float)
ts_p_diff = (ts_p-ts_p.shift()).dropna(how='any')
ts_r = df['redemption'].astype(float)
ts_r_diff = (ts_r-ts_r.shift()).dropna(how='any')
ts_r_diff2 = (ts_r_diff-ts_r_diff.shift()).dropna(how='any')

#print ts_p

'''
plt.plot(ts_p_diff)
plt.plot(ts_r_diff)
plt.legend()
plt.show()
'''

#test_stationarity(ts_r_diff,12)
#sgt.plot_acf(ts_r_diff, lags=140)
#sgt.plot_pacf(ts_r_diff, lags=40)  #, lags=40
#plt.show()


'''
# 循环寻找 ARIMA 参数  p q
p=0;q=0;min_rss=float("inf") 
for i in range(0,10):
    print i
    for j in range(0,10):
        try:
            model = ARIMA(ts_r, (i,1,j))
            results = model.fit(disp=-1)
            rss = sum((results.fittedvalues-ts_r_diff)**2)
            print rss
            if rss<min_rss:   # 0阶：ts  1阶：ts_diff
                min_rss = rss     # 0阶：ts  1阶：ts_diff
                p=i;q=j;
        except:
            pass
print p,q,min_rss
'''


# 购买预测
purchaseModel = ARIMA(ts_p, (7,1,5)).fit(disp=-1)
purchasePredict_diff = purchaseModel.predict('2014-08-31', '2014-09-30')
purchasePredict_diff_cumsum = purchasePredict_diff.cumsum() 
purchasePredict = pd.Series(ts_p.ix[-1], index=pd.date_range('2014-08-31', '2014-09-30', freq='1d'))
purchasePredict = purchasePredict.add(purchasePredict_diff_cumsum,fill_value=0) 
purchasePredict.astype(integer).to_csv('D:\\Documents\\SQL\\tianchi\\profile_predict\\purchasePredict.csv')
'''
plt.figure(1)
plt.plot(ts_p, color='r')
plt.plot(purchasePredict, color='b')
'''

# 赎回预测
redemptionModel = ARIMA(ts_r, (8,1,7)).fit(disp=-1)
redemptionPredict_diff = redemptionModel.predict('2014-08-31', '2014-09-30')
redemptionPredict_diff_cumsum = redemptionPredict_diff.cumsum() 
redemptionPredict = pd.Series(ts_r.ix[-1], index=pd.date_range('2014-08-31', '2014-09-30', freq='1d'))
redemptionPredict = redemptionPredict.add(redemptionPredict_diff_cumsum,fill_value=0) 
redemptionPredict.astype(integer).to_csv('D:\\Documents\\SQL\\tianchi\\profile_predict\\redemptionPredict.csv')
'''
plt.figure(2)
plt.plot(ts_r, color='r')
plt.plot(redemptionPredict, color='b')
plt.show()
'''

predict_table = pd.DataFrame([ 20140900+i for i in range(1,31)])
predict_table['p'] = purchasePredict.astype(integer).values[1:]
predict_table['r'] = redemptionPredict.astype(integer).values[1:]
predict_table.to_csv('D:\\Documents\\SQL\\tianchi\\profile_predict\\tc_comp_predict_table.csv', index=False, header=False)

print 'finished'