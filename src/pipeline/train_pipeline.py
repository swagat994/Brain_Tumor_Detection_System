from src.components.data_ingestion import DataIngestion

from src.components.data_transformation import (
    DataTransformation
)

from src.components.model_trainer import (
    ModelTrainer
)


class TrainPipeline:

    def run_pipeline(self):

        print(
            "Starting Data Ingestion..."
        )

        ingestion = DataIngestion()

        ingestion.initiate_data_ingestion()

        print(
            "Data Ingestion Completed"
        )

        print(
            "Starting Data Transformation..."
        )

        transformation = DataTransformation()

        (
            train_generator,
            validation_generator,
            class_indices
        ) = transformation.get_data_generators()

        print(
            "Data Transformation Completed"
        )

        print(
            "Starting Model Training..."
        )

        trainer = ModelTrainer()

        trainer.train_model(
            train_generator=train_generator,
            validation_generator=validation_generator,
            num_classes=len(class_indices)
        )

        print(
            "Model Training Completed"
        )


if __name__ == "__main__":

    pipeline = TrainPipeline()

    pipeline.run_pipeline()