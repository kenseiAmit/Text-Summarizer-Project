from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.config.configurations import ConfigurationManger


class DataTransformationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self) -> None:
        config = ConfigurationManger()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(data_transformation_config)
        data_transformation.convert()
        