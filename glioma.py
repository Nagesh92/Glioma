import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,PrecisionRecallDisplay

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

data = pd.read_csv('Glioma.csv',sep=',')
data

df = pd.DataFrame(data)
df

df.head(5)

df.tail(5)

df.info()

df.describe()

df.dtypes

df.corr()

sns.heatmap(df.corr(),annot=True)
plt.show()

df.columns

df['Grade'].value_counts()

sns.countplot(data=df,x = 'Grade')
plt.show()

x = df.drop(['Grade'],axis=1)
y = df.Grade

x_train,x_test,y_train,y_test = tts(x,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

lr = LogisticRegression()
lr.fit(x_train,y_train.ravel())
y_pred = lr.predict(x_test)
acc_lr = accuracy_score(y_test,y_pred)
print(acc_lr)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

PrecisionRecallDisplay.from_estimator(lr,x_test,y_test)
plt.show()

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
pred_rf = rf.predict(x_test)
acc_rf = accuracy_score(y_test,pred_rf)
print(acc_rf)
print(confusion_matrix(y_test,pred_rf))
print(classification_report(y_test,pred_rf))

PrecisionRecallDisplay.from_estimator(rf,x_test,y_test)
plt.show()


pickle.dump(lr,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
model

