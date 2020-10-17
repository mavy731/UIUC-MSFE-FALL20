#YUE MA (yuema4)
#FALL20
#IE517 Group Project CreditScore
#Textbook:Python Machine Learning_Sebastain Raschka_2nd

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV


#Data processing and preparation
df=pd.read_csv('MLF_GP1_CreditScore.csv')
print(df.head())
Rating=df.Rating.unique()
print(Rating)
le=LabelEncoder()
stdsc=StandardScaler()
X=df.drop(['InvGrd','Rating'],axis=1).values
X=stdsc.fit_transform(X)
y_grade=df['InvGrd'].values
y_rating=le.fit_transform(df['Rating'].values)
print(y_rating)
x_train,x_test, y_grade_train,y_grade_test=train_test_split(X,y_grade,test_size=0.15, random_state=1)
x_train,x_test, y_rating_train,y_rating_test=train_test_split(X,y_rating,test_size=0.15, random_state=1)

#Feature Subtraction PCA Binary Classification
cov_mat=np.cov(x_train.T)
eigen_vals, eigen_vecs=np.linalg.eig(cov_mat)

tot=sum(eigen_vals)
var_exp=[(i/tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp=np.cumsum(var_exp)
print('sum',cum_var_exp[:15])
plt.bar(range(1,26),var_exp[:25],alpha=0.5,align='center',label='Individual explained variance')
plt.step(range(1,26),cum_var_exp[:25],where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.savefig('Explained variance_pca.png',dpi=300)
plt.show()

pca=PCA(n_components=15)
x_grade_train=pca.fit_transform(x_train)
x_grade_test=pca.transform(x_test)
x_combined = np.vstack((x_grade_train, x_grade_test))
y_combined = np.hstack((y_grade_train, y_grade_test))

#Feature Selection_PCA_Multiclass
pva=PCA(n_components=15)
x_rating_train=pca.fit_transform(x_train,y_rating_train)
x_rating_test=pca.transform(x_test)

#Evaluation function
def evaluation_print(classifier,x_train,y_train,x_test,y_test,label):
    classifier.fit(x_train,y_train)
    y_train_pred = classifier.predict(x_train)
    y_test_pred = classifier.predict(x_test)
    score1 = accuracy_score(y_train, y_train_pred)
    score2 = accuracy_score(y_test, y_test_pred)
    print('Accuracy scores (in-sample): ', np.round(score1, decimals=5))
    print('Accuracy scores (out-sample): ', np.round(score2, decimals=5))
    scores = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10, n_jobs=1)
    print(label + ' cv acciracy scores: %s' % scores)
    print(label + 'CV accurracy: %.5f +/- %.5f' % (np.mean(scores), np.std(scores)))


#Binomial Classification_logreg
lr=LogisticRegression(C=100,random_state=1)
lr.fit(x_grade_train,y_grade_train)
y_grade_pred=lr.predict(x_grade_test)
score=lr.score(x_grade_test,y_grade_test)
print('Binomial Classification logreg')
print('Accuracy: %.3f' % score)
evaluation_print(lr,x_grade_train,y_grade_train,x_grade_test,y_grade_test,'DecisionTree')
train_sizes, train_scores, test_scores = learning_curve(LogisticRegression(C=100,random_state=1),
                                                        x_grade_train,
                                                        y_grade_train,
                                                        train_sizes=[10,50,80,120,200,300,400],cv=5)
label="Grade logreg"
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5, 1.00])
plt.tight_layout()
name = label + '_learning_curve.png'
plt.figtext(0.5, 0.01,label , wrap=True, horizontalalignment='center', fontsize=12)
plt.savefig(name, dpi=300)
plt.show()

#Binomial Classification_KNN
knn = KNeighborsClassifier(n_neighbors=8,
                           metric='minkowski')
knn.fit(x_grade_train, y_grade_train)
score=knn.score(x_grade_test,y_grade_test)
print('Binomial Classification knn')
print('Accuracy: %.3f' % score)
evaluation_print(knn,x_grade_train,y_grade_train,x_grade_test,y_grade_test,'knn')
train_sizes, train_scores, test_scores = learning_curve(knn,
                                                        x_grade_train,
                                                        y_grade_train,
                                                        train_sizes=[10,50,80,120,200,300,400],cv=5)
label="Grade knn"
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5, 1.00])
plt.tight_layout()
name = label + '_learning_curve.png'
plt.figtext(0.5, 0.01,label , wrap=True, horizontalalignment='center', fontsize=12)
plt.savefig(name, dpi=300)
plt.show()

#Binomial Classification_DecisionTree
tree=DecisionTreeClassifier(criterion='gini',max_depth=10,random_state=1)
tree.fit(x_grade_train,y_grade_train)
score=tree.score(x_grade_test,y_grade_test)
print('Binomial Classification decision tree')
print('Accuracy: %.3f' % score)
evaluation_print(tree,x_grade_train,y_grade_train,x_grade_test,y_grade_test,'DecisionTree')
train_sizes, train_scores, test_scores = learning_curve(tree,
                                                        x_grade_train,
                                                        y_grade_train,
                                                        train_sizes=[10,50,80,120,200,300,400],cv=5)
label="Grade Decision Tree"
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5, 1.00])
plt.tight_layout()
name = label + '_learning_curve.png'
plt.figtext(0.5, 0.01,label , wrap=True, horizontalalignment='center', fontsize=12)
plt.savefig(name, dpi=300)
plt.show()

#Multiclass Classifictaion_KNN

knn = KNeighborsClassifier(n_neighbors=8,
                           metric='minkowski')
knn.fit(x_rating_train, y_rating_train)
score=knn.score(x_rating_test,y_rating_test)
print('Multiclass Classification knn')
print('Accuracy: %.3f' % score)
evaluation_print(knn,x_rating_train,y_rating_train,x_rating_test,y_rating_test,'knn')

#Multiclass Classification_DecisonTree
tree=DecisionTreeClassifier(criterion='gini',max_depth=10,random_state=42)
tree.fit(x_rating_train,y_rating_train)
score=tree.score(x_rating_test,y_rating_test)
print('Multiclass Classification decision tree')
print('Accuracy: %.3f' % score)
evaluation_print(tree,x_rating_train,y_rating_train,x_rating_test,y_rating_test,'DecisionTree')

#Find best k
k_range=range(1,26)
scores=[]
scores_tst=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_grade_train,y_grade_train)
    y_pred=knn.predict(x_grade_test)
    scores.append(accuracy_score(y_grade_test,y_pred))
    scores_tst.append(knn.score(x_grade_train,y_grade_train))


plt.plot(range(1,26),scores,'r-',label="Train")
plt.plot(range(1,26),scores_tst,'b-',label='Test')
plt.xlabel('Neighbors K')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('KNN_OptK_TS.png',dpi=300)
plt.show()

k_range=range(1,26)
scores=[]
scores_tst=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_rating_train,y_rating_train)
    y_pred=knn.predict(x_rating_test)
    scores.append(accuracy_score(y_rating_test,y_pred))
    scores_tst.append(knn.score(x_rating_train,y_rating_train))


plt.plot(range(1,26),scores,'r-',label="Train")
plt.plot(range(1,26),scores_tst,'b-',label='Test')
plt.xlabel('Neighbors K')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('KNN_OptK_TS.png',dpi=300)
plt.show()

tree=DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
param_range=[1,2,3,4,5,6,7,8,9,10,11,12]
param_grid=[{'n_neighbors':param_range,
                           'p':[2],
                           'metric':['minkowski']}]
gs=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=param_grid,scoring='accuracy',
                cv=10,
                n_jobs=-1)
gs.fit(x_rating_train,y_rating_train)
print(gs.best_score_)
print(gs.best_params_)
gs.fit(x_grade_train,y_grade_train)
print(gs.best_score_)
print(gs.best_params_)

param_grid=[{'max_depth':param_range}]
gs=GridSearchCV(estimator=DecisionTreeClassifier(criterion='gini',random_state=1),param_grid=param_grid,scoring='accuracy',
                cv=10,
                n_jobs=-1)
gs.fit(x_rating_train,y_rating_train)
print(gs.best_score_)
print(gs.best_params_)
gs.fit(x_grade_train,y_grade_train)
print(gs.best_score_)
print(gs.best_params_)