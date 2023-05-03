from numpy import *
from pandas import *
import pandas as pd
from matplotlib.pyplot import *
from seaborn import *
from sklearn.linear_model import LinearRegression
data=read_csv("C:\\Users\\KIIT\\Downloads\\boston-housing-dataset.csv")
print(data.corr())
figure(figsize=(10,6))
heatmap(data.corr(),annot=True,linewidth=2)
l_column=list(data.columns)
len_feature=len(l_column)
print(l_column)
print(len_feature)
X= data[l_column[0:len_feature-1]]
y = data[l_column[len_feature-1]]
from sklearn.model_selection import train_test_split
from sklearn import metrics
lm=LinearRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)
lm.fit(X_train,y_train)
y_pred=lm.predict(X_test)
print(y_pred)