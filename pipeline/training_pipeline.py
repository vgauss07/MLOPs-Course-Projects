from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import CONFIG_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH
from config.paths_config import PROCESSED_DIR, PROCESSED_TRAIN_DATA_PATH
from config.paths_config import PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH


if __name__ == '__main__':
    # 1. data ingestion
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    # 2. data preprocesing
    processor = DataProcessor(TRAIN_FILE_PATH,
                              TEST_FILE_PATH,
                              PROCESSED_DIR,
                              CONFIG_PATH)
    processor.process()

    # 3. Model Training
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH,
                            PROCESSED_TEST_DATA_PATH,
                            MODEL_OUTPUT_PATH)
    trainer.run()
