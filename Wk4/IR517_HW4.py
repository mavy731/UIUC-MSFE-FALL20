#YUE MA (yuema4)
#FALL20
#IE517 HW4
#Textbook:Python Machine Learning_Sebastain Raschka_2nd

import pandas as pd
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


#read housing data
df1=pd.read_csv('housing.csv')
df1s=df1.describe(include='all').T
df1s.round(3)
df1s.to_csv('df1s.csv',index=True)
print(df1.describe(include='all'))
print(df1.head())
df2=pd.read_csv('housing2.csv')
print(df2.describe(include='all'))
print(df2.head())

#Exploratory Data Analysis
cols=df1.drop(['CHAS'],axis=1).columns.values
sns.pairplot(df1[cols], height=2.5)
plt.tight_layout()
plt.savefig("Pairplot.png",dpi=300)
plt.show()

#Pearson R and headtmap
cols=df1.columns.values
cm=np.corrcoef(df1.values.T)
sns.set(font_scale=0.7)
htmp=sns.heatmap(cm,cbar=True, annot=True,square=True, fmt='.2f',xticklabels=cols,yticklabels=cols)
plt.tight_layout()
plt.savefig('heatmap.png',dpi=300)
plt.show()

#Prepare Data
X=df1.drop(['MEDV'],axis=1).values
y=df1['MEDV'].values
x_train,x_test, y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)
reg=LinearRegression()


#Linear Regression
reg.fit(x_train,y_train)
y_train_pred=reg.predict(x_train)
y_test_pred=reg.predict(x_test)
plt.scatter(y_train_pred,y_train_pred-y_train,c='steelblue',marker='o',label='Trainning data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='limegreen',marker='s',label='Testing data')
plt.xlabel('Predicted Values')
plt.ylabel('Rediduals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=1)
plt.xlim([-10,50])
plt.savefig('linearresidual.png',dpi=300)
plt.show()
print('Coefficient:\n',reg.coef_)
print('Intercept: \n',reg.intercept_)
print('MSE train: %.3f, test: % .3f'%(mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
print('R^2 train: % .3f, test: % .3f' %(r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))


#Prepare Data2
X=df2.drop(['MEDV'],axis=1).values
y=df2['MEDV'].values
x_train,x_test, y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)
alpha=[0.1, 0.4, 0.7, 1]

#Ridge
for a in alpha:
    ridge=Ridge(alpha=a)
    ridge.fit(x_train,y_train)
    y_train_pred = ridge.predict(x_train)
    y_test_pred = ridge.predict(x_test)
    plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', label='Trainning data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', label='Testing data')
    title='Ridge Alpha='+str(a)
    plt.title(title)
    plt.xlabel('Predicted Values')
    plt.ylabel('Rediduals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=1)
    plt.xlim([-10, 50])
    name='ridge'+str(a)+'.png'
    plt.savefig(name, dpi=300)
    plt.show()
    print('Ridge alpha=',a)
    print('Coefficient:\n', ridge.coef_)
    print('Intercept: \n', ridge.intercept_)
    print('MSE train: %.3f, test: % .3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: % .3f, test: % .3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

#Lasso
for a in alpha:
    lasso=Lasso(alpha=a)
    lasso.fit(x_train,y_train)
    y_train_pred = lasso.predict(x_train)
    y_test_pred = lasso.predict(x_test)
    plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', label='Trainning data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', label='Testing data')
    title = 'Lasso Alpha=' + str(a)
    plt.title(title)
    plt.xlabel('Predicted Values')
    plt.ylabel('Rediduals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=1)
    plt.xlim([-10, 50])
    name = 'lasso' + str(a) + '.png'
    plt.savefig(name, dpi=300)
    plt.show()
    print('Ridge alpha=', a)
    print('Coefficient:\n', lasso.coef_)
    print('Intercept: \n', lasso.intercept_)
    print('MSE train: %.3f, test: % .3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: % .3f, test: % .3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


#ElasticNet
for a in alpha:
    elanet=ElasticNet(alpha=a,l1_ratio=0.5)
    elanet.fit(x_train,y_train)
    y_train_pred = elanet.predict(x_train)
    y_test_pred = elanet.predict(x_test)
    plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', label='Trainning data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', label='Testing data')
    title = 'ELasticNet Alpha=' + str(a)
    plt.title(title)
    plt.xlabel('Predicted Values')
    plt.ylabel('Rediduals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=1)
    plt.xlim([-10, 50])
    name = 'elasticnet' + str(a) + '.png'
    plt.savefig(name, dpi=300)
    plt.show()
    print('Ridge alpha=', a)
    print('Coefficient:\n', elanet.coef_)
    print('Intercept: \n', elanet.intercept_)
    print('MSE train: %.3f, test: % .3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: % .3f, test: % .3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


print("My name is Yue Ma")
print("My NetID is: yuema4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")