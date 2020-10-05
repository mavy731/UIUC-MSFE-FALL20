#YUE MA (yuema4)
#FALL20
#IE517 HW6
#Textbook:Python Machine Learning_Sebastain Raschka_2nd

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#read data
df1=pd.read_csv('ccdefault.csv')
df=df1.drop(['ID'],axis=1)
y=df['DEFAULT']
X=df.drop(['DEFAULT'],axis=1)

#

def train_and_valuation(Tree, x_train, y_train,n):
    Tree.fit(x_train, y_train)
    y_train_pred = Tree.predict(x_train)
    y_test_pred=Tree.predict(X_test)
    score1=accuracy_score(y_train,y_train_pred)
    score2=accuracy_score(y_test,y_test_pred)
    scores.append(score1)
    scores_tst.append(score2)


#Accuracy
RS_range = range(1, 11)
scores = []
scores_tst = []
for n in RS_range:
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.1, random_state=n, stratify=y)
    Tree = DecisionTreeClassifier()
    train_and_valuation(Tree,X_train, y_train,n)

print('Accuracy scores (in-sample): ',np.round(scores,decimals=5))
print('Accuracy scores (out-sample): ',np.round(scores_tst,decimals=5))

print('In-sample accuracy scores: %.5f +/- %.5f' %(np.mean(scores), np.std(scores)))
print('Out-sample accuracy scores: %.5f +/- %.5f' %(np.mean(scores_tst), np.std(scores_tst)))

#KFolds
scores = []
for n in RS_range:
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.1, random_state=n, stratify=y)
    Tree = DecisionTreeClassifier()
    scores=cross_val_score(estimator=Tree, X=X_train,y=y_train,cv=10, n_jobs=1)
    print('Randonm state: ',n)
    print('CV accuracy scores:', np.round(scores,decimals=5))
    print('CV accurracy: %.5f +/- %.5f' %(np.mean(scores),np.std(scores)))

print("My name is Yue Ma")
print("My NetID is: yuema4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")