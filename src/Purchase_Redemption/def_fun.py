from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def test_stationarity(timeseries,w): 
    #Determing rolling statistics 
    rolmean = timeseries.rolling(window=w,center=False).mean()
    rolstd = timeseries.rolling(window=w,center=False).std()
    #Plot rolling statistics: 
    plt.subplot(211)
    plt.plot(timeseries, color='blue',label='Original') 
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.legend(loc='best')
    plt.title('Original & Rolling Mean') 
    plt.subplot(212) 
    plt.plot(rolstd, color='black', label = 'Rolling Std') 
    plt.legend(loc='best') 
    plt.title('Standard Deviation') 
    plt.show() 
    #Perform Dickey-Fuller test: 
    print 'Results of Dickey-Fuller Test:' 
    dftest = adfuller(timeseries, autolag='AIC') 
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used']) 
    for key,value in dftest[4].items(): 
        dfoutput['Critical Value (%s)'%key] = value 
    print dfoutput
    