# importing the libraries
import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
# importing deom the .py files
from src.exception import CustomException
# defining the function
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
# evaluating the models 

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_) # setting up the best parameters
            model.fit(X_train,y_train) # fitting the model 

            #model.fit(X_train, y_train)  # Train model
 
            y_train_pred = model.predict(X_train) # predicting the model

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred) # evaluating the r2 square

            test_model_score = r2_score(y_test, y_test_pred) # evaluating the r2 square

            report[list(models.keys())[i]] = test_model_score  # creating the report 

        return report 

    except Exception as e:
        raise CustomException(e, sys)