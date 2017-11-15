# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# read data
X_train_path = 'D:/Blue/HKUST/BDT/6000B Deep Learning/Project/traindata.csv'
y_train_path = 'D:/Blue/HKUST/BDT/6000B Deep Learning/Project/trainlabel.csv'
X_test_path = 'D:/Blue/HKUST/BDT/6000B Deep Learning/Project/testdata.csv'
X = pd.read_csv(X_train_path ,header = None, encoding = 'utf-8')
y = pd.read_csv(y_train_path ,header = None, encoding = 'utf-8')
X_train = np.array(X)
y_train = np.array(y).flatten()
X_t = pd.read_csv(X_test_path ,header = None, encoding = 'utf-8')
X_test = np.array(X_t)

# default model
RFModel = RandomForestClassifier(oob_score = True)
RFModel.fit(X_train,y_train)
print("default---Out-of-bag score: ", RFModel.oob_score_)

# find the best n_esitmators   
param_treeNum = {'n_estimators':range(100,401,10)}
Gsearch1 = GridSearchCV(estimator = RandomForestClassifier(max_depth = None, 
                                      max_features = 'sqrt',random_state = 304),
                        param_grid = param_treeNum, scoring = 'roc_auc',cv = 5)
Gsearch1.fit(X_train,y_train)
print(Gsearch1.best_params_,Gsearch1.best_score_)
# 210

# consruct a new model by the best parameter
RFModel2 = RandomForestClassifier(n_estimators = Gsearch1.best_params_, max_depth = None,
                                  max_features = 'sqrt',random_state = 300,
                                  oob_score = True)
# calculate oob score
RFModel2.fit(X_train,y_train)
print("default---Out-of-bag score: ", RFModel2.oob_score_)

# auc score
y_train_pred = RFModel2.predict_proba(X_train)[:,1]
print("AUC Score(train): %f" %roc_auc_score(y_train,y_train_pred))

# use new model to predict the test data
x_test_pred = RFModel2.predict(X_test)
label_DF = pd.DataFrame(x_test_pred)
label_path = 'D:/Blue/HKUST/BDT/6000B Deep Learning/Project/project1_20446054.csv'
label_DF.to_csv(label_path, index = False, header = None, encoding = 'utf-8')