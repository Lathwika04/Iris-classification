import pandas as pd

dataset=pd.read_csv("iris.data.csv")


dataset["type"].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2],inplace=True)
print(dataset)

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
y_pred_train=model.predict(x_train)

from sklearn.metrics import confusion_matrix,accuracy_score
cmt=confusion_matrix(y_train,y_pred_train)
print(cmt)
print(accuracy_score(y_train,y_pred_train)*100)
cm=confusion_matrix(y_test,y_pred)
print(cm)
acc=accuracy_score(y_test,y_pred)
print(acc*100)
