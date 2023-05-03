# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:46:54 2023

@author: KIIT
"""

from pandas import *
from numpy import *
from matplotlib.pyplot import *
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from seaborn import *
from sklearn.linear_model import LinearRegression
df=read_csv("C:\\Users\\KIIT\\Downloads\\Fish.csv")
print(df)
print(df.columns)
l_col=list(df.columns)
print(l_col)
l_len=len(l_col)
print(l_len)
X=df[l_col[2:]]
y=df[l_col[1]]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=123)
m=LinearRegression()
print(m)
m.fit(X_train,y_train)
pred=m.predict(X_test)
print(pred)
