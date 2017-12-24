"""
@author: MPatel

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


## Class to implement various classification methods on the data:
### While initiating the class, it requires three arguments:
### 1. predictors: Numpy-matrix of predictors
### 1. outcome: Numpy-array of true-labels
### 3. test_frac: Proportion of test fraction
 
class SupervisedClassificationModels:
    
    def __init__(self, predictors, outcome, test_frac): 
        self._predictors = predictors
        self._outcome = outcome
        self._test_frac = test_frac
            
    def encode_features():    ## Yet to be written
        pass  
    
    def create_dummies():   ## Yet to be written
        pass       
        
    def fit_logistic_regression(self):
        """
        Fit a Logistic-Regression model on the data.
        
        Args: None
        Return: 
            lr: Logistic-Regression object
            cm: Confusion-Matrix
        
        """
        
        X_train, X_test, y_train, y_test = train_test_split(self._predictors, 
                                                            self._outcome,
                                                            test_size = self._test_frac,
                                                            random_state = 0)
        
        ## Feature Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        ##Fitting a model:
        lr = LogisticRegression(random_state = 101)
        lr = lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        cm = confusion_matrix(y_pred = y_pred, y_true = y_test)
       
#        print (classification_report(y_pred = y_pred, y_true = y_test))
        
        return lr, cm

    ## Customize the random-forest object by allowing users to provide extra set of arguments
    ## in this function:
    def fit_random_forest(self):
        """
        Fit a Random-Forest model on the data.
        
        Args: None
        Return: 
            rf: Random-Forest object
            cm: Confusion-Matrix
        
        """
        
        X_train, X_test, y_train, y_test = train_test_split(self._predictors, 
                                                            self._outcome, 
                                                            test_size = self._test_frac, 
                                                            random_state = 0)
        
        ## Feature Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        ##Fitting a model:
        rf = RandomForestClassifier(random_state = 101)
        rf = rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        cm = confusion_matrix(y_pred = y_pred, y_true = y_test)
       
#        print (classification_report(y_pred = y_pred, y_true = y_test))
        
        return rf, cm


    def fit_support_vector_classifier(self):
        """
        Fit a Support-Vector-Classifier on the data.
        
        Args: None
        Return: 
            sv: Support-Vector-Classifier object
            cm: Confusion-Matrix
        
        """
        
        X_train, X_test, y_train, y_test = train_test_split(self._predictors, 
                                                            self._outcome, 
                                                            test_size = self._test_frac, 
                                                            random_state = 0)
        
        ## Feature Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        ##Fitting a model:
        sv = svm.SVC(random_state = 101)
        sv = sv.fit(X_train, y_train)
        y_pred = sv.predict(X_test)
        cm = confusion_matrix(y_pred = y_pred, y_true = y_test)
       
#        print (classification_report(y_pred = y_pred, y_true = y_test))
        
        return sv, cm

