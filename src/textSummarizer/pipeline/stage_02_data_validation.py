from textSummarizer.components.data_validation import DataValidation
from textSummarizer.config.configurations import ConfigurationManger


class DataValidationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self) -> None:
        config = ConfigurationManger()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(data_validation_config)
        data_validation.validate_all_files_exist()