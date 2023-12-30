from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.config.configurations import ConfigurationManger


class ModelTrainerPipeline:
    def __init__(self) -> None:
        pass

    def main(self) -> None:
        config = ConfigurationManger()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()