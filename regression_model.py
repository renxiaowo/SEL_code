# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 15:52:21 2018
@author: renyi
"""
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.linear_model import  LinearRegression
from sklearn import metrics
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_classif

X=np.loadtxt('F:\\work\\predict_economics\\data\\model\\AggregatedData.txt')
y=np.loadtxt('F:\\work\\predict_economics\\data\gdp\\Gdp.txt') 
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape


R2=[]
idx=[]
for i in range(2024,2025):
    sk = SelectKBest(f_classif, k=i)
    X_3=sk.fit_transform(X, y)
#model = LinearRegression()
    print i
    model = LinearRegression()
    predicted = cross_val_predict(model, X_3, y, cv=5)

    r2=r2_score(y, predicted)
    R2.append(r2)
    idx.append(i)
    print 'R2:',r2
    print "MSE:",metrics.mean_squared_error(y, predicted)
    print "RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted))

fig, ax = plt.subplots()
N=len(predicted)
s = (8*np.random.rand(N))**2
ax.scatter(y, predicted,alpha=0.6,s=s)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k-', lw=1)
ax.set_xlabel('Actual GDP',fontsize=12)
ax.set_ylabel('Predicted GDP',fontsize=12)
#plt.savefig('F:\\work\\predict_economics\\jpg2eps\\RF_regression.eps',format='eps', dpi=300)





