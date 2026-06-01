import os

from tensorflow.keras.applications import EfficientNetB0 # type: ignore

from tensorflow.keras.layers import ( # type: ignore
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.models import Model # type: ignore

from tensorflow.keras.optimizers import Adam # type: ignore

from tensorflow.keras.callbacks import ( # type: ignore
    EarlyStopping,
    ModelCheckpoint
)


class ModelTrainer:

    def __init__(self):

        self.model_dir = "models"

        os.makedirs(
            self.model_dir,
            exist_ok=True
        )

    def build_model(
        self,
        num_classes
    ):

        base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

        # Stage 1 Transfer Learning
        base_model.trainable = False

        x = base_model.output

        x = GlobalAveragePooling2D()(x)

        x = Dropout(0.4)(x)

        x = Dense(
            256,
            activation="relu"
        )(x)

        x = Dropout(0.3)(x)

        predictions = Dense(
            num_classes,
            activation="softmax"
        )(x)

        model = Model(
            inputs=base_model.input,
            outputs=predictions
        )

        model.compile(
            optimizer=Adam(
                learning_rate=1e-3
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def train_model(
        self,
        train_generator,
        validation_generator,
        num_classes
    ):

        model = self.build_model(
            num_classes=num_classes
        )

        model.summary()

        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        )

        checkpoint = ModelCheckpoint(
            filepath="models/best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )

        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=10,
            callbacks=[
                early_stopping,
                checkpoint
            ]
        )

        model.save(
            "models/final_model.keras"
        )

        return history