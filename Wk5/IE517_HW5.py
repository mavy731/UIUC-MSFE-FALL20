#YUE MA (yuema4)
#FALL20
#IE517 HW5
#Textbook:Python Machine Learning_Sebastain Raschka_2nd

import pandas as pd
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.decomposition import PCA

#read data
df1=pd.read_csv('hw5_treasury yield curve data.csv')
print(df1.head())
df1=df1.drop(['Date'],axis=1)
print(df1.head())

df1s=df1.describe(include='all').T
df1s.columns=['count','mean','std','min','25','50','75','max']
print(df1s.head())
df1s=df1s.round(2)
df2 = df1s.rename({'25%': '25', '50%': '50','75%':'75'}, axis='columns')
df1s.to_csv('df1s.csv',index=True)
#Exploratory Data Analysis
#cols=['SVENF01','SVENF05','SVENF10','SVENF15','SVENF20','SVENF25','SVENF30','Adj_Close']
#sns.pairplot(df1[cols], height=2.5)
#plt.tight_layout()
#plt.savefig("Pairplot.png",dpi=300)
#plt.show()

#Pearson R and headtmap
#df=df1[cols]
#cm=np.corrcoef(df.values.T)
#sns.set(font_scale=0.7)
#htmp=sns.heatmap(cm,cbar=True, annot=True,square=True, fmt='.2f',xticklabels=cols,yticklabels=cols)
#plt.tight_layout()
#plt.savefig('heatmap.png',dpi=300)
#plt.show()

#Prepare Data
X=df1.drop(['Adj_Close'],axis=1).values
y=df1['Adj_Close'].values
x_train,x_test, y_train,y_test=train_test_split(X,y,test_size=0.15, random_state=42)
scalerX=StandardScaler().fit(x_train)
x_train=scalerX.transform(x_train)
x_test=scalerX.transform(x_test)

#valuation function
def train_and_valuation(clf,x_train,y_train):
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    print ("Coefficient of determination on training set", clf.score(x_train,y_train))
    print("Coefficient of determination on testing set", clf.score(x_test, y_test))
    print('MSE train: %.3f, test: % .3f' % (
    mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))

#print model
def print_model(clf):
    print('Coefficient:\n', clf.coef_)
    print('Intercept: \n', clf.intercept_)

#LinearRegrssion and SVN regressor
clf_sgf=linear_model.SGDRegressor(loss='squared_loss',penalty=None, random_state=42)
train_and_valuation(clf_sgf,x_train,y_train)
print_model(clf_sgf)

slf_svr=svm.SVR(kernel='linear')
train_and_valuation(slf_svr,x_train,y_train)
print_model(slf_svr)

#PCA
x_train,x_test, y_train,y_test=train_test_split(X,y,test_size=0.15, random_state=42)
sc=StandardScaler()
x_train_std=sc.fit_transform(x_train)
x_test_std=sc.transform(x_test)
cov_mat=np.cov(x_train_std.T)
eigen_vals, eigen_vecs=np.linalg.eig(cov_mat)

tot=sum(eigen_vals)
var_exp=[(i/tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp=np.cumsum(var_exp)
print('sum',cum_var_exp[:3])

plt.bar(range(1,11),var_exp[:10],alpha=0.5,align='center',label='Individual explained variance')
plt.step(range(1,11),cum_var_exp[:10],where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.savefig('Explained variance.png',dpi=300)
plt.show()

pca=PCA(n_components=3)
x_train_pca=pca.fit_transform(x_train_std)
x_test_pca=pca.transform(x_test_std)

clf_sgf_pca=linear_model.SGDRegressor(loss='squared_loss',penalty=None, random_state=42)
clf_sgf_pca.fit(x_train_pca,y_train)
y_train_pred = clf_sgf_pca.predict(x_train_pca)
y_test_pred = clf_sgf_pca.predict(x_test_pca)
print("Coefficient of determination on training set", clf_sgf_pca.score(x_train_pca, y_train))
print("Coefficient of determination on testing set", clf_sgf_pca.score(x_test_pca, y_test))
print('MSE train: %.3f, test: % .3f' % (
    mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))

print_model(clf_sgf_pca)

print("slf_svr_pca")
slf_svr=svm.SVR(kernel='linear')
slf_svr.fit(x_train_pca,y_train)
y_train_pred = slf_svr.predict(x_train_pca)
y_test_pred = slf_svr.predict(x_test_pca)
print("Coefficient of determination on training set", slf_svr.score(x_train_pca, y_train))
print("Coefficient of determination on testing set", slf_svr.score(x_test_pca, y_test))
print('MSE train: %.3f, test: % .3f' % (
    mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))

print_model(slf_svr)

print("My name is Yue Ma")
print("My NetID is: yuema4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")