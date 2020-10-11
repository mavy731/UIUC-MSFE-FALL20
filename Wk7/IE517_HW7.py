#YUE MA (yuema4)
#FALL20
#IE517 HW7
#Textbook:Python Machine Learning_Sebastain Raschka_2nd

from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

#read and prepare data
df1=pd.read_csv('ccdefault.csv')
df=df1.drop(['ID'],axis=1)
y=df['DEFAULT']
X=df.drop(['DEFAULT'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1)
#Random Forest
test_score=[]
grid={'n_estimators':[5,10,20,50,100,200],'max_depth':[3],'max_features':[4]}
rf=RandomForestClassifier()
labels=['5','10','20','50','100','200']
print('10-fold cross validation:\n')
for g, label in zip(ParameterGrid(grid),labels):
    rf.set_params(**g)
    score=cross_val_score(estimator=rf,X=X_train,y=y_train,cv=10,scoring='roc_auc')
    print("ROC AUC: %0.2F (+/- %0.2f) [%s]" %(score.mean(), score.std(), label))

#Feature importance
grid={'n_estimators':[20],'max_depth':[3],'max_features':[4]}
rf.set_params(**g)
rf.fit(X_train,y_train)
feature_names=X.columns
importances=rf.feature_importances_
sorted_index=np.argsort(importances[::-1])
n=range(len(importances))
labels=np.array(feature_names)[sorted_index]

plt.bar(n,importances[sorted_index],tick_label=labels)
plt.xticks(rotation=90)
plt.savefig("Importance.png",dpi=300)
plt.show()


print("My name is Yue Ma")
print("My NetID is: yuema4")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")