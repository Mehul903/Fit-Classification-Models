# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 14:42:55 2017

@author: MPatel
"""

import SupervisedModels as sm
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# =============================================================================
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# 
# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, classification_report
# 
# =============================================================================


## Importing the dataset
dataset = pd.read_csv('../data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  ## Removing unnecessary columns
y = dataset.iloc[:, 13].values

### Remove one of the dummy columns of country variable to avoid dummy variable trap:
#X = X[:, 1:]


## Fitting models using my module!
LR = sm.SupervisedClassificationModels(X, y, 0.2, [1,2])
lr, cm = LR.fit_logistic_regression()

RF = sm.SupervisedClassificationModels(X, y, 0.2, [1,2])
rf, cm = RF.fit_random_forest()

SV = sm.SupervisedClassificationModels(X, y, 0.2, [1,2])
sv, cm = SV.fit_support_vector_classifier()