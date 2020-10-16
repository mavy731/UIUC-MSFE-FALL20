#YUE MA (yuema4)
#FALL20
#IE517 Group Project CreditScore
#Textbook:Python Machine Learning_Sebastain Raschka_2nd

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')

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
x_train,x_test, y_grade_train,y_grade_test=train_test_split(X,y_grade,test_size=0.15, random_state=42)
x_train,x_test, y_rating_train,y_rating_test=train_test_split(X,y_rating,test_size=0.15, random_state=42)

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

#Feature Selection_LDA_Multiclass
lda=LDA(n_components=15)
x_rating_train=lda.fit_transform(x_train,y_rating_train)
x_rating_test=lda.transform(x_test)

#Evaluation function
def evaluation_print(classifier,x,y,label):
    scores=cross_val_score(estimator=classifier, X=x,y=y,cv=10,n_jobs=1)
    print(label+' cv acciracu scores: %s'%scores)
    print(label+ ' cv accuracy: %')
    print('CV accurracy: %.5f +/- %.5f' % (np.mean(scores), np.std(scores)))

#Learning Curve
def learning_curve(classifier,x,y,label):
    train_sizes, train_scores, test_scores = learning_curve(estimator=classifier,
                                                            X=x,
                                                            y=y,
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            cv=10,
                                                            n_jobs=1)

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
    plt.ylim([0.8, 1.03])
    plt.tight_layout()
    name=label+'_learning_curve.png'
    plt.savefig(name, dpi=300)
    plt.show()


#Binomial Classification_logreg
lr=LogisticRegression(C=100,random_state=1)
lr.fit(x_grade_train,y_grade_train)
y_grade_pred=lr.predict(x_grade_test)
score=lr.score(x_grade_test,y_grade_test)
print('Binomial Classification logreg')
print('Accuracy: %.3f' % score)
evaluation_print(lr,x_grade_train,y_grade_train,'DecisionTree')
#learning_curve(lr,x_grade_train,y_grade_train,'DecisionTree')

#Binomial Classification_KNN
knn = KNeighborsClassifier(n_neighbors=5,
                           p=2,
                           metric='minkowski')
knn.fit(x_grade_train, y_grade_train)
score=knn.score(x_grade_test,y_grade_test)
print('Binomial Classification knn')
print('Accuracy: %.3f' % score)
evaluation_print(knn,x_grade_train,y_grade_train,'knn')

#Binomial Classification_DecisionTree
tree=DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
tree.fit(x_grade_train,y_grade_train)
score=tree.score(x_grade_test,y_grade_test)
print('Binomial Classification decision tree')
print('Accuracy: %.3f' % score)
evaluation_print(tree,x_grade_train,y_grade_train,'DecisionTree')

#Multiclass Classifictaion_KNN

knn = KNeighborsClassifier(n_neighbors=9,
                           p=2,
                           metric='minkowski')
knn.fit(x_rating_train, y_rating_train)
score=knn.score(x_rating_test,y_rating_test)
print('Multiclass Classification knn')
print('Accuracy: %.3f' % score)
evaluation_print(knn,x_grade_train,y_grade_train,'knn')

#Multiclass Classification_DecisonTree
tree=DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
tree.fit(x_rating_train,y_rating_train)
score=tree.score(x_rating_test,y_rating_test)
print('Multiclass Classification decision tree')
print('Accuracy: %.3f' % score)
evaluation_print(tree,x_grade_train,y_grade_train,'DecisionTree')
