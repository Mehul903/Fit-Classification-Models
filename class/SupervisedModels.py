"""
@author: MPatel

"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support



class SupervisedClassificationModels:
    
    """
    Need to change this docstring.
    
    Fits multiple supervised classification models on the data.
    
    Args:
        preedictors (numpy-matrix): Matrix of predictor variables (or features)
        outcome (numpy-array): Array of outcome (or target variable)
        test_frac (float): Fraction of data to be considered as test set
        col_ind (list): List containing indices of columns of categorical-type
        class_repot (boolean): Default 'False', if 'True' then classification-report is 
                                saved in a dataframe.
    
    """
    
    def __init__(self, predictors, outcome, test_frac, col_ind, 
                 class_report = False, matrix = False): 
        self._predictors = predictors
        self._outcome = outcome
        self._test_frac = test_frac
        self._col_ind = col_ind    
        self._class_report = class_report
        self._matrix = matrix
        
    def _feature_engineering(self):
        """
        Convert categorical features (including binary features as well) into labels and 
        then create dummies of those features.
        
        Args: None
        Return: Transformed version of predictors
        
        """
    
        if self._matrix:
            self._predictors[:, self._col_ind] = np.apply_along_axis(lambda col: LabelEncoder().fit_transform(col), 0, self._predictors[:, self._col_ind])
            self._predictors = OneHotEncoder(categorical_features = [self._col_ind], sparse = False).fit_transform(self._predictors)    
           
            return self._predictors
        
        else:    
            cat_data = self._predictors.iloc[:, self._col_ind]
            
            self._predictors.drop(self._predictors.columns[self._col_ind], axis = 1, inplace = True)
            original_col_names = list(self._predictors.columns)
            

            dummy_col_names = []
            for col in self._col_ind:
                LE = LabelEncoder()
                cat_data.iloc[:, col-1] = LE.fit_transform(list(cat_data.iloc[:, col-1]))
                
                new_names_temp = [LE.classes_[i]+'_'+str(i) for i in range(len(LE.classes_))]    
                dummy_col_names.extend(new_names_temp)
            
    
            OHE = OneHotEncoder(sparse = False)
            cat_data = OHE.fit_transform(cat_data)    
            cat_data = pd.DataFrame(cat_data, columns = dummy_col_names)
    
            self._predictors = pd.concat([self._predictors, cat_data], axis = 1)
#            final_col_names = []
#            final_col_names.extend(original_col_names)
#            final_col_names.extend(dummy_col_names)
    
    
            return self._predictors    

    
    
    def _train_test_split(self):
        """
        Split the data into train-test set for further modeling process. Returns a tuple of 
        train and test set.
        
        Args:
            None
            
        Return:
            X_train (): Matrix of predictors-train set
            X_test (): Matrix of predictors-test set
            y_train (): Array of outcome-train set
            y_test (): Array of outcome-test set            
        
        """
        
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
       
        p,r,f,s = precision_recall_fscore_support(y_true = y_test, y_pred = y_pred)
        
        if self._class_report:
            
            classification_report = pd.DataFrame({'Precision': p, 
                                                  'Recall': r, 
                                                  'F_score': f, 
                                                  'Support': s, 
                                                  'Class': np.unique(y_test)})
            return lr, cm, classification_report
        
        return lr, cm


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
       
        p,r,f,s = precision_recall_fscore_support(y_true = y_test, y_pred = y_pred)
        
        if self._class_report:
            
            classification_report = pd.DataFrame({'Precision': p, 
                                                  'Recall': r, 
                                                  'F_score': f, 
                                                  'Support': s, 
                                                  'Class': np.unique(y_test)})
            return rf, cm, classification_report 
        
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
       
        p,r,f,s = precision_recall_fscore_support(y_true = y_test, y_pred = y_pred)
        
        if self._class_report:
            
            classification_report = pd.DataFrame({'Precision': p, 
                                                  'Recall': r, 
                                                  'F_score': f, 
                                                  'Support': s, 
                                                  'Class': np.unique(y_test)})
            return sv, cm, classification_report
        
        return sv, cm



## Importing the dataset
dataset = pd.read_csv('../data/Churn_Modelling.csv')
#X = dataset.iloc[:, 3:13]   ## Removing unnecessary columns
#y = dataset.iloc[:, 13]
  
X = dataset.iloc[:, 3:13].values  ## Removing unnecessary columns
y = dataset.iloc[:, 13].values



## Merge above dataframe with original dataframe:
#X = pd.concat([X, X_cat_data], axis = 1)
#final_col_names = []
#final_col_names.extend(col_names)
#final_col_names.extend(new_col_names)
  

LR = SupervisedClassificationModels(predictors = X, outcome = y, 
                                      test_frac = 0.2, col_ind = [1,2], 
                                      class_report = True, matrix=True)
lr, cm, cr = LR.fit_logistic_regression()
  

RF = SupervisedClassificationModels(X, y, 0.2, [1,2], class_report = True, matrix = True)
rf, cm, cr = RF.fit_random_forest()


SV = SupervisedClassificationModels(X, y, 0.2, [1,2], class_report = True, matrix = True)
SV, cm, cr = SV.fit_support_vector_classifier()
  
## Remove one of the dummy columns of country variable to avoid dummy variable trap:
#X = X[:, 1:]
        
    