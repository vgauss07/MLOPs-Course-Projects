import pandas as pd
import os
import yaml

from src.custom_exception import CustomException
from src.logger import get_logger


logger = get_logger(__name__)


# Read the YAML file
def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError('File not in path')

        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info('Successfully read the YAML file')
            return config
    except Exception:
        logger.error("Error while reading YAML file")
        raise CustomException("Failed to read YAML file")


def load_data(path):
    try:
        logger.info("Loading data")
        return pd.read_csv(path)
    except Exception as ce:
        logger.error(f"error loading data: {ce}")
        raise CustomException(f"Failed to load data {ce}")
