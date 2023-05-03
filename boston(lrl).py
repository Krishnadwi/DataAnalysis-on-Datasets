# -*- coding: utf-8 -*-
"""
Created on Mon May  1 23:16:42 2023

@author: KIIT
"""
from pandas import *
from sklearn.datasets import load_boston
from numpy import *
from matplotlib.pyplot import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from seaborn import *
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
df=load_boston()
print(df)
ds=DataFrame(df.data)
print(ds)
ds.columns=df.feature_names
X=ds
y=df.target
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
m=LinearRegression()
m.fit(X_train,y_train)
pred=m.predict(X_test)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.model_selection import cross_val_score
mse=cross_val_score(m,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
print(mean(mse))
pred=m.predict(X_test)
print(displot(pred-y_test,kind='kde'))
ridge=Ridge()
p={'alpha':[1,2,3,4,5,6,7,8,9,10,11]}
cv=GridSearchCV(ridge,p,scoring='neg_mean_squared_error',cv=5)
ridge.fit(X_train,y_train)
r_pred=ridge.predict(X_test)
mse=cross_val_score(ridge,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
print(mean(mse))
lasso=Lasso()
a={'alpha':[1,2,3,4,5,6,7,8,9,10,11]}
gcv=GridSearchCV(lasso,p,scoring='neg_mean_squared_error',cv=5)
lasso.fit(X_train,y_train)
r_pred=ridge.predict(X_test)
mse=cross_val_score(lasso,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
print(mean(mse))

