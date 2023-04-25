'''
It is the process of converting the categorical variables into numerical and scaling the numerical values.
'''
# Importing the libraries.
import sys 
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('Artifacts', 'preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig
    def get_data_transformer_obj(self):
        """
        This is the function which is responsible for the transformation the features.
        """
        
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity','parental_level_of_education',
                'lunch','test_preparation_course'
            ]
            num_pipeline = Pipeline(
                ('Imputer', SimpleImputer(strategy= 'median')),
                ('Scaler', StandardScaler()))
            cat_pipeline = Pipeline(
                steps=[
                ('Imputer', SimpleImputer(strategy='most_frequent')),
                ('Encoding', OneHotEncoder()),
                ('Scaler', StandardScaler())
                ])
            logging.info('Numerical columns scaling completed.')
            logging.info('Categoical columns Encoding completed.')


            preprocessor = ColumnTransformer(
                [
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )


            return preprocessor
        except Exception as e :
            raise CustomException(e,sys)
            
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Reading the training and testing data is completed!')
            logging.info('obtaining the preprocessing object.')
            preprocessing_obj = self.get_data_transformer_obj()
            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop([target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop([target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying the preprocessing on training and testing data!')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(' Saved the preprocessing object !')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path, 
                obj = preprocessing_obj
            )
            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
        

          
        except Exception as e:
            raise CustomException(e, sys)