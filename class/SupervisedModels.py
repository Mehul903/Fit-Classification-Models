"""
@author: MPatel

"""

import numpy as np

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
    
    def __init__(self, predictors, outcome, test_frac, col_ind): 
        self._predictors = predictors
        self._outcome = outcome
        self._test_frac = test_frac
        self._col_ind = col_ind           
        
    def _feature_engineering(self):
        """
        Convert categorical features (including binary features as well) into labels and 
        then create dummies of those features.
        
        Args: None
        Return: Transformed version of predictors
        
        """
    
        self._predictors[:, self._col_ind] = np.apply_along_axis(lambda col: LabelEncoder().fit_transform(col), 0, self._predictors[:, self._col_ind])
        self._predictors = OneHotEncoder(categorical_features = [self._col_ind]).fit_transform(self._predictors)    
    
        return self._predictors.toarray()
    
    def _train_test_split(self):
        
        X_train, X_test, y_train, y_test = train_test_split(self._predictors, 
                                                            self._outcome,
                                                            test_size = self._test_frac,
                                                            random_state = 0)
        
        
        return X_train, X_test, y_train, y_test
    
    def fit_logistic_regression(self):
        """
        Fit a Logistic-Regression model on the data.
        
        Args: None
        Return: 
            lr: Logistic-Regression object
            cm: Confusion-Matrix
        
        """
        
        self._predictors = self._feature_engineering()
        
        X_train, X_test, y_train, y_test = self._train_test_split()

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
        
        self._predictors = self._feature_engineering()
        
        X_train, X_test, y_train, y_test = self._train_test_split()
        
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
        
        self._predictors = self._feature_engineering()
        
        X_train, X_test, y_train, y_test = self._train_test_split()
        
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



# =============================================================================
# ## Importing the dataset
# dataset = pd.read_csv('../data/Churn_Modelling.csv')
# X = dataset.iloc[:, 3:13].values  ## Removing unnecessary columns
# y = dataset.iloc[:, 13].values
# 
# LR = SupervisedClassificationModels(predictors = X, outcome = y, 
#                                     test_frac = 0.2, col_ind = [1,2])
# lr, cm = LR.fit_logistic_regression()
# 
# RF = SupervisedClassificationModels(X, y, 0.2, [1,2])
# rf, cm = RF.fit_random_forest()
# 
# SV = SupervisedClassificationModels(X, y, 0.2, [1,2])
# SV, cm = RF.fit_support_vector_classifier()
# 
# ## Remove one of the dummy columns of country variable to avoid dummy variable trap:
# X = X[:, 1:]
# 
# =============================================================================


    
    
    
    