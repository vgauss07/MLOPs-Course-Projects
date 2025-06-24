import numpy as np
import os
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from config.paths_config import *
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml, load_data


logger = get_logger(__name__)


class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self, df):
        try:
            logger.info("Starting Data Processing")

            logger.info("Dropping the columns")
            df.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']

            logger.info("Applying Label Encoder")

            le = LabelEncoder()
            mappings = {}

            for col in cat_cols:
                df[col] = le.fit_transform(df[col])
                mappings[col] = {label: code for label,
                                 code in zip(le.classes_,
                                             le.transform(le.classes_))}

            logger.info("Label Mappings are : ")
            for col, mapping in mappings.items():
                logger.info(f"{col} : {mapping}")

            logger.info("Performing Skewness Handling")

            skew_threshold = self.config['data_processing']['skewness_threshold']  # wq:noqa
            skewness = df[num_cols].apply(lambda x: x.skew())

            for column in skewness[skewness > skew_threshold].index:
                df[column] = np.log1p(df[column])

            return df

        except Exception as e:
            logger.error(f"Error during preprocess step, {e}")
            raise CustomException(f"Error while preprocessing data, {e}")

    def balance_data(self, df):
        try:
            logger.info("Handling the Imbalanced Data")

            X = df.drop(columns='booking_status')
            y = df['booking_status']

            smote = SMOTE(random_state=42)

            X_resampled, y_resampled = smote.fit_resample(X, y)

            balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_data['booking_status'] = y_resampled

            logger.info("Data balanced successfully")
            return balanced_data

        except Exception as e:
            logger.error(f"Error during Balancing data step {e}")
            raise CustomException(f"error while balancing data, {e}")

    def feature_selection(self, df):
        try:
            logger.info("Starting Feature Selection Step")

            X = df.drop(columns='booking_status')
            y = df['booking_status']

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            feature_importance = model.feature_importances_

            feature_importance_df = pd.DataFrame({'feature': X.columns,
                                                  'importance': feature_importance})

            top_feature_importance_df = feature_importance_df.sort_values(by='importance',
                                                                          ascending=False)

            num_features_to_select = self.config['data_processing']['no_of_features']

            top_10_features = top_feature_importance_df['feature'].head(num_features_to_select).values

            logger.info(f'Features Selected: {top_10_features}')

            top_10_df = df[top_10_features.tolist() + ["booking_status"]]

            logger.info("Feature Selection Completed")

            return top_10_df

        except Exception as e:
            logger.error(f'Error during feature selection step {e}')
            raise CustomException(f'Error while selecting features {e}')

    def save_data(self, df, file_path):
        try:
            logger.info("Saving processed data in processed dir")

            df.to_csv(file_path, index=False)

            logger.info(f'data saved successfully to {file_path}')

        except Exception as e:
            logger.error(f'Error during saving data step {e}')
            raise CustomException('Error while saving data', e)

    def process(self):
        try:
            logger.info("Loading data from RAW directory")
            
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.feature_selection(train_df)
            test_df = test_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info('Data Processing Completed Successfully')

        except Exception as e:
            logger.error(f'Error during preprocessing pipeline, {e}')
            raise CustomException("Error while preprocessing pipeline,", e)


if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()
