# -*- coding: utf-8 -*-
"""
Created on Sat Jun 02 13:46:11 2018
SVM CV=5
class_weight='balanced'
@author: renyi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn import metrics
from heapq import nlargest
#data&label
X1=np.loadtxt('F:\\work\\predict_economics\\data\\model\\AggregatedData.txt')
y_try=np.array((1,1,1,1,1,3,3,3,3,3),dtype=int)
y1=np.loadtxt('F:\\work\\predict_economics\\data\\model\\GdpLabel3class.txt')#3 SEL
y3=np.loadtxt('F:\\work\\predict_economics\\data\\model\\GdpLabel4class.txt')#4 SEL
sc = StandardScaler()
X2= sc.fit_transform(X1)

idx=[]
score1_idx=[]
score2_idx=[]
score3_idx=[]
for i in range(1,2000,30):
    X_3=SelectKBest(f_classif, k=i).fit_transform(X2, y1)   #3 SEL
    X_4 = SelectKBest(f_classif, k=i).fit_transform(X2, y3) #4 SEL
    #3 SEL
    clf1 = SVC(decision_function_shape='ovo',class_weight='balanced',kernel='linear')    
    predicted1 = cross_val_predict(clf1, X_3, y1, cv=5).astype(int)
    acc1=metrics.accuracy_score(y1, predicted1)
    idx.append(i)
    score1_idx.append(acc1)

    #4 SEL
    clf3 = SVC(decision_function_shape='ovo',class_weight='balanced',kernel='linear')    
    predicted3 = cross_val_predict(clf3, X_4, y3, cv=5).astype(int)
    acc3=metrics.accuracy_score(y3, predicted3)
    score3_idx.append(acc3)
    
plt.grid(True)
ax = plt.gca()
ax.yaxis.grid(True)
ax.xaxis.grid(False)
plt.plot(idx,score1_idx,label='3-class',color='grey',linewidth=2, linestyle="-")
plt.scatter([381, ], [0.866, ], s=50, color='blue',marker='^')
plt.plot(idx,score3_idx,label='4-class',color='darkorange',linewidth=2, linestyle="--")
plt.scatter([1081, ], [0.727, ], s=50, color='blue',marker='^')
plt.xlabel('Number of features',fontsize=15)
plt.ylabel('Accuracy',fontsize=15)
plt.legend(fontsize=12)
plt.show
