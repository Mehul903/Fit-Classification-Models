"""
@author: MPatel

"""

import SupervisedModels as sm
import pandas as pd
import numpy as np



### If input data is in numpy-matrix form:
## Importing the dataset
dataset = pd.read_csv('../data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  ## Removing unnecessary columns
y = dataset.iloc[:, 13].values

## Fitting models using my module!
LR = sm.SupervisedClassificationModels(predictors = X, outcome = y, 
                                       test_frac = 0.2, col_ind = [1,2], 
                                       class_report = True, matrix = True)
lr, cm, cr = LR.fit_logistic_regression()
print ('Confusion matrix for Logistic-Regression: \n', cm)
print ('Classification report for Logistic-Regression: \n', cr)


RF = sm.SupervisedClassificationModels(X, y, 0.2, [1,2], class_report = True, matrix = True)
rf, cm, cr = RF.fit_random_forest()
print ('Confusion matrix for Random-Forest: \n', cm)
print ('Classification report for Random-Forest: \n', cr)


SV = sm.SupervisedClassificationModels(X, y, 0.2, [1,2], class_report = True, matrix = True)
SV, cm, cr = SV.fit_support_vector_classifier()
print ('Confusion matrix for Support-Vector-Classifier: \n', cm)
print ('Classification report for Support-Vector-Classifier: \n', cr)



### If input data is dataframe:
## Importing the dataset
dataset = pd.read_csv('../data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]  ## Removing unnecessary columns
y = dataset.iloc[:, 13]

## Fitting models using my module!
LR = sm.SupervisedClassificationModels(predictors = X, outcome = y, 
                                       test_frac = 0.2, col_ind = [1,2], 
                                       class_report = True)
lr, cm, cr = LR.fit_logistic_regression()
print ('Confusion matrix for Logistic-Regression: \n', cm)
print ('Classification report for Logistic-Regression: \n', cr)


RF = sm.SupervisedClassificationModels(predictors = X, outcome = y, 
                                       test_frac = 0.2, col_ind = [1,2], 
                                       class_report = True)
rf, cm, cr = RF.fit_random_forest()
print ('Confusion matrix for Random-Forest: \n', cm)
print ('Classification report for Random-Forest: \n', cr)


SV = sm.SupervisedClassificationModels(predictors = X, outcome = y, 
                                       test_frac = 0.2, col_ind = [1,2], 
                                       class_report = True)
SV, cm, cr = SV.fit_support_vector_classifier()
print ('Confusion matrix for Support-Vector-Classifier: \n', cm)
print ('Classification report for Support-Vector-Classifier: \n', cr)

















