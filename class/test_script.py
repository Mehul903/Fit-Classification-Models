# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 14:42:55 2017

@author: MPatel
"""

import SupervisedModels as sm
import pandas as pd
import numpy as np


## Importing the dataset
dataset = pd.read_csv('../data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  ## Removing unnecessary columns
y = dataset.iloc[:, 13].values

### Remove one of the dummy columns of country variable to avoid dummy variable trap:
#X = X[:, 1:]


## Fitting models using my module!
LR = SupervisedClassificationModels(predictors = X, outcome = y, 
                                    test_frac = 0.2, col_ind = [1,2], 
                                    class_report = True)
lr, cm, cr = LR.fit_logistic_regression()

RF = SupervisedClassificationModels(X, y, 0.2, [1,2], class_report = True)
rf, cm, cr = RF.fit_random_forest()

SV = SupervisedClassificationModels(X, y, 0.2, [1,2], class_report = True)
SV, cm, cr = RF.fit_support_vector_classifier()