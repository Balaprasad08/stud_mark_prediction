import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('student_info.csv')
df.head()
df.shape
sns.scatterplot(df['study_hours'],df['student_marks'],color='Red')
plt.xlabel('Study Hours')
plt.ylabel('Student Marks')
plt.show()
df.isnull().sum()
df['study_hours']=df['study_hours'].fillna(df['study_hours'].mean())
df['study_hours'].isnull().sum()
df.isnull().sum()
df.head()
X=df.iloc[:,:-1]
X.head()
y=df.iloc[:,-1]
y.head()
from sklearn.model_selection import train_test_split,cross_val_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
def check_model(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print('r2_score:',r2_score(y_test,y_pred))
check_model(LinearRegression(),X_train,X_test,y_train,y_test)
check_model(RandomForestRegressor(),X_train,X_test,y_train,y_test)
check_model(DecisionTreeRegressor(),X_train,X_test,y_train,y_test)
check_model(SVR(),X_train,X_test,y_train,y_test)
check_model(KNeighborsRegressor(),X_train,X_test,y_train,y_test)
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
print('r2_score: ',r2_score(y_test,y_pred))
print('MSE: ',mean_squared_error(y_test,y_pred))
print('MAE: ',mean_absolute_error(y_test,y_pred))
print('RMSE: ',np.sqrt(mean_squared_error(y_test,y_pred)))
lr.score(X_test,y_test)
lr.score(X_train,y_train)
cv=cross_val_score(LinearRegression(),X_train,y_train,cv=5)
print('n_split: ',cv)
print('Average: ',np.average(cv))
sns.distplot(y_test-y_pred)
sns.scatterplot(y_test,y_pred)
lr.predict([[12]])
data=np.array([8]).reshape(1,-1)
lr.predict(data)
import pickle
import joblib
pickle.dump(lr,open('Stu_mark.pickle','wb'))
joblib.dump(lr,'Stu_mark.joblib')
model_pickle=pickle.load(open('Stu_mark.pickle','rb'))
prediction=model_pickle.predict(X_test)
r2_score(y_test,prediction)
model_pickle.score(X_test,y_test)
model_pickle.score(X_train,y_train)
model_joblib=joblib.load('Stu_mark.joblib')
y_job_pred=model_joblib.predict(X_test)
r2_score(y_test,y_job_pred)
model_joblib.score(X_test,y_test)
model_joblib.score(X_train,y_train)    