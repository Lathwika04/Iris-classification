import pandas as pd

dataset=pd.read_csv("iris.data.csv")

dataset["type"].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2],inplace=True)

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

print(dataset)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)

print(y_test)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test,y_pred)
print(cm)
a=accuracy_score(y_pred,y_test)
print(a*100)