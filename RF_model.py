# -*- coding: utf-8 -*-
"""
Created on Mon Jun 04 22:14:04 2018
@author: renyi
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import metrics
import matplotlib.pyplot as plt
#data&label
X=np.loadtxt('F:\\work\\predict_economics\\data\\model\\AggregatedData.txt')
y1=np.loadtxt('F:\\work\\predict_economics\\data\\model\\GdpLabel3class.txt') #3 SEL
y3=np.loadtxt('F:\\work\\predict_economics\\data\\model\\GdpLabel3class.txt') #4 SEL

#RF
idx=[]
score1_idx=[]
score2_idx=[]
score3_idx=[]
for i in range(1,2000,30):
    print i
    #3 SEL
    X_3=SelectKBest(f_classif, k=i).fit_transform(X, y1) #3 SEL
    X_4 = SelectKBest(f_classif, k=i).fit_transform(X,y3) #4 SEL
    clf1 = RandomForestClassifier(class_weight='balanced',n_estimators=83, max_depth=9, \
                               random_state=51)
    predicted1 = cross_val_predict(clf1, X_3, y1, cv=5).astype(int)
    acc1=metrics.accuracy_score(y1, predicted1)
    idx.append(i)
    score1_idx.append(acc1)


    #4 SEL
    clf3 = RandomForestClassifier(class_weight='balanced',n_estimators=9, max_depth=9, \
                               random_state=51)

    predicted3 = cross_val_predict(clf3, X_4, y3, cv=5).astype(int)
    acc3=metrics.accuracy_score(y3, predicted3)
    score3_idx.append(acc3)
    
plt.grid(True)
ax = plt.gca()
ax.yaxis.grid(True)
ax.xaxis.grid(False)
plt.plot(idx,score1_idx,label='3-class',color='grey',linewidth=2, linestyle="-")
plt.scatter([31, ], [0.845, ], s=50, color='blue',marker='^')
plt.plot(idx,score3_idx,label='4-class',color='darkorange',linewidth=2, linestyle="--")
plt.scatter([211, ], [0.657, ], s=50, color='blue',marker='^')
plt.xlabel('Number of features',fontsize=15)
plt.ylabel('Accuracy',fontsize=15)
plt.legend(fontsize=12)
plt.show