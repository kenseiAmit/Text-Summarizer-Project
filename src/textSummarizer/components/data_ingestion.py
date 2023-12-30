import os
from pathlib import Path
from urllib import request
from textSummarizer.entity import DataIngestionConfig
from textSummarizer.utils.common import get_size
from textSummarizer.logging import logger
import zipfile

class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config=config
    
    def download_file(self) -> None:
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} downloaded with the following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self) -> None:
        """
        zip_file_path: str
        Extract the zip file into the data directory
        Function returns none
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)