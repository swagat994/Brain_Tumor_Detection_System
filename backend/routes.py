import os

from fastapi import (
    APIRouter,
    UploadFile,
    File
)

from fastapi.responses import FileResponse

from src.pipeline.predict_pipeline import (
    PredictPipeline
)

from src.components.gradcam import (
    GradCAM
)

router = APIRouter()


@router.post("/predict")
async def predict_image(
    file: UploadFile = File(...)
):

    temp_path = f"temp_{file.filename}"

    with open(
        temp_path,
        "wb"
    ) as f:

        f.write(
            await file.read()
        )

    predictor = PredictPipeline()

    result = predictor.predict(
        temp_path
    )

    os.remove(
        temp_path
    )

    return result


@router.post("/gradcam")
async def generate_gradcam(
    file: UploadFile = File(...)
):

    try:

        input_path = f"temp_{file.filename}"

        output_path = (
            f"gradcam_{file.filename}"
        )

        with open(
            input_path,
            "wb"
        ) as f:

            f.write(
                await file.read()
            )

        gradcam = GradCAM()

        gradcam.save_gradcam(
            image_path=input_path,
            output_path=output_path
        )

        os.remove(
            input_path
        )

        return FileResponse(
            output_path,
            media_type="image/jpeg"
        )

    except Exception as e:

        return {
            "error": str(e)
        }