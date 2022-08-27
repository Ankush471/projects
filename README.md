# projects
Taitanic Data Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv('../input/titanic/train.csv')
df.head()
df.shape
df.columns
df.describe()
df.describe(include='O') 
df.isnull().sum()
df.columns
dummies=pd.get_dummies(df, columns=['Sex','Embarked'], drop_first=True)
plt.figure(figsize=(50,7))
sns.pairplot(data=df,kind='scatter',hue='Survived')
plt.show()plt.figure(figsize=(15,7))
for i in conti_data:
    
    sns.histplot(data=df,x=i,hue='Survived',kde=True,bins=40)
    plt.show()
plt.figure(figsize=(25,7))
cor=df.corr()
sns.heatmap(cor,annot=True, linewidths=0.3)
plt.figure(figsize=(25,7))
cor=df.corr()
sns.heatmap(cor,annot=True, linewidths=0.3)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.80)
from sklearn.preprocessing import StandardScaler
sclar=StandardScaler()
x_train=sclar.fit_transform(x_train)
x_test=sclar.fit_transform(x_test)
from sklearn.linear_model import LogisticRegression
algo=LogisticRegression()
algo.fit(x_train,y_train)
a=algo.predict(x_train)
b=algo.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,b)
print(cm)
Accuracy of Logistic Regression
from sklearn.metrics import accuracy_score
train_score=accuracy_score(a,y_train)
print('accuracy of Logi_Regg train dataset is:',train_score)
test_score=accuracy_score(b,y_test)
print('Accuracy of Logi_Regg test dataset is:',test_score)
Accuracy Support Vector Machine
from sklearn.svm import SVC
svm=SVC()
svm.fit(x_train,y_train)
z=svm.score(x_test,y_test)
print('accuracy of SVM:',z)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
d=dt.score(x_test,y_test)
print('Accuracy of DecisionTree:',d)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
e=rfc.score(x_test,y_test)
print('Accuracy of RandomForest:',e)
