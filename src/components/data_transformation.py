from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

from tensorflow.keras.applications.efficientnet import (  # type: ignore
    preprocess_input
)


class DataTransformation:

    def __init__(self):

        self.train_path = "artifacts/raw/Training"

        self.test_path = "artifacts/raw/Testing"

    def get_data_generators(self):

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

        train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            shuffle=True
        )

        validation_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            shuffle=False
        )

        return (
            train_generator,
            validation_generator,
            train_generator.class_indices
        )