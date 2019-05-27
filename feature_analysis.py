# -*- coding: utf-8 -*-
"""
Created on Mon May 07 21:52:59 2018
@author: renyi
"""
import numpy as np
import matplotlib.pyplot as plt
feature_score1=np.loadtxt('F:\\work\\predict_economics\\data\\model\\rf_top_feature_v2.txt')
block=np.loadtxt('F:\\work\\predict_economics\\data\\gdp\\gdp2block.txt')
feature=np.loadtxt('F:\\work\\predict_economics\\data\\model\\feature_new_3.0.txt')
feature_score2=np.unique(feature_score1, axis=0)
feature_score=feature_score2[np.lexsort(-feature_score2.T)] 
List=list(feature_score[:,0])
#block
gdp1=block[:,[0,3]]
gdp=np.delete(gdp1,[4],axis=0)

top=feature_score[:41] 
top_id=top[:,0].astype(int)
poi=top[top[:,0]<21,:]
trace1=top[top[:,0]<53,:]
trace=trace1[trace1[:,0]>20,:]
#app1=top[top[:,0]<309,:]
#app=app1[app1[:,0]>52,:]
#od=top[top[:,0]>308,:]
app1=top[top[:,0]<69,:]
app=app1[app1[:,0]>52,:]
od=top[top[:,0]>68,:]


feature_top=feature[:,top_id] 
feature1=np.hstack((gdp,feature_top))
feature2=feature1[np.argsort(-feature1[:,1])]
feature_ana=feature2[[0,1,2,3,4,182,183,184,185,186],:]

#poi
poi_idx=[] #poi
for item in poi[:,0]:
    idx=List.index(item)
    poi_idx.append(idx)
poi_ana=np.column_stack((poi_idx,poi))

#trace_feature
trace_idx=[] #trace
for item in trace[:,0]:
    idx=List.index(item)
    trace_idx.append(idx)
trace_ana=np.column_stack((trace_idx,trace))

#app_feature
app_idx=[] #app
t_idx=[] #app
app_class=[]#app
for item in app[:,0]:
    idx1=List.index(item)
    app_idx.append(idx1)
    t=int((item-52)/16)+1
    t_idx.append(t)
    app_id=(item-52)
    app_class.append(app_id)
app_ana1=np.column_stack((app_idx,app))
app_ana2=np.column_stack((app_ana1,t_idx))
app_ana=np.column_stack((app_ana2,app_class))
























