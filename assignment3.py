
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn import tree
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics

df=pd.read_csv("sonar.all-data",header=None)
df[60] = df[60].apply(lambda x: 0 if x=='R' else 1)
y=df[60]
x=df.drop(60,axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


def decesiontreeclassifier():
    parameters = [{'max_depth':[2,4,6,8,10],'min_samples_split':[2,3,4,5],'min_samples_leaf':[2,3,4,5]}]
    clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=parameters,n_jobs=4,cv=5)
    clf.fit(X=x_train, y=y_train)
    tree_model = clf.best_estimator_
    print (clf.best_score_, clf.best_params_)
    pred = clf.predict(x_test)
    rf_probs1= clf.predict_proba(x_test)[:, 1]
    print("roc value:", roc_auc_score(y_test, pred))
    print(f'training Model Accuracy: {clf.score(x_train, y_train)}')
    #test error
    print(metrics.accuracy_score(y_test,pred))
    
decesiontreeclassifier()


# In[88]:


def randomforestclassifier():
    param_grid = {'bootstrap': [True,False],'max_depth': [2,3,4,5,6,7],'min_samples_leaf': [2,3,4,5, 6],
    'min_samples_split': [2,3,4,5,6,7,8],'n_estimators': [10,100,500,1000]}
    rfc=RandomForestClassifier(random_state=42,max_features='sqrt')
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(x_train, y_train)
    best_params=CV_rfc.best_params_
    print(best_params)
    rfc1=RandomForestClassifier(random_state=42,min_samples_leaf=best_params['min_samples_leaf'],max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],max_features='sqrt',n_estimators= 100, criterion=['gini'] )   
    rfc1.fit(x_train, y_train)
    pred=rfc1.predict(x_test)
    print("Accuracy for Random Forest on CV data: ",metrics.accuracy_score(y_test,pred))
    
randomforestclassifier() 


# In[56]:


from sklearn import svm, grid_search
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 1.1,2,3,10]
    gammas = [0.001, 0.01, 0.1, 1]
    #kernels = [‘linear’, ‘rbf’, ‘poly’]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search,grid_search.best_params_


grid_search,params=svc_param_selection(x_train, y_train, 10)
y_pred=grid_search.predict(x_test)
print('best param:',params)
kernals="linear,rbf,poly"
kernals=kernals.split(',')
for kernel in kernals:
    svc=SVC(kernel=kernel,C=params['C'],gamma=params['gamma'])
    svc.fit(x_train,y_train)
    y_pred=svc.predict(x_test)
    print("kernal name:",kernel)
    print('Accuracy Score:')
    print(metrics.accuracy_score(y_test,y_pred))
    

