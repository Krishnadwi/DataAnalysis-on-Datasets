from numpy import *
from pandas import *
from matplotlib.pyplot import *
from sklearn.model_selection import train_test_split
from seaborn import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
de=read_csv("C:\\Users\\KIIT\\Documents\\ML notes\\iris (1) - Copy.csv")
print(de)
print(de.info(verbose=True))
print(de.describe(percentiles=[0.1,0.25,0.5,0.7,0.9]))
print(de.columns)
print(pairplot(de))
l_column=list(de.columns)
l_feature=len(l_column)
X=de[l_column[0:l_feature-1]]
y=de[l_column[l_feature-2]]
for i in de['class']:
    if i == "Iris-setosa":
        de['class']=de['class'].replace("Iris-setosa",'1')
    elif i == "Iris-virginica":
        de['class']=de['class'].replace("Iris-virginica",'2')
    elif i == "Iris-versicolor":
        de['class']=de['class'].replace('Iris-versicolor','3')
print(de)
de['class']=de['class'].astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)
a=LinearRegression()
a.fit(X_train,y_train)
y_pred=a.predict(X_test)
print(y_pred)

