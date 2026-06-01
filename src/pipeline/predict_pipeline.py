import numpy as np

from tensorflow.keras.models import load_model # type: ignore

from tensorflow.keras.preprocessing import image # type: ignore

from tensorflow.keras.applications.efficientnet import ( # type: ignore
    preprocess_input
)


class PredictPipeline:

    def __init__(self):

        self.model = load_model(
            "models/best_model.keras"
        )

        self.class_names = {
            0: "glioma",
            1: "meningioma",
            2: "notumor",
            3: "pituitary"
        }

    def predict(self, image_path):

        img = image.load_img(
            image_path,
            target_size=(224, 224)
        )

        img_array = image.img_to_array(
            img
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        img_array = preprocess_input(
            img_array
        )

        prediction = self.model.predict(
            img_array,
            verbose=0
        )

        predicted_index = np.argmax(
            prediction
        )

        confidence = float(
            np.max(prediction)
        ) * 100

        predicted_class = self.class_names[
            predicted_index
        ]

        tumor_detected = (
            predicted_class != "notumor"
        )

        result = {

            "tumor_detected": tumor_detected,

            "tumor_type": predicted_class,

            "confidence": round(
                confidence,
                2
            )
        }

        return result