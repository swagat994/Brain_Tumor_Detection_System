# MRI Brain Tumor Detection and Classification System

An end-to-end deep learning system for detecting and classifying brain tumors from MRI scans using **EfficientNetB0 and transfer learning**. The system performs four-class classification, provides prediction confidence, and uses **Grad-CAM** to visualize regions that influence model predictions.

## Overview

The MRI Brain Tumor Detection and Classification System is a deep learning-based computer vision project designed to classify brain MRI scans into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

The project follows a modular machine learning architecture covering data ingestion, image preprocessing, data augmentation, model training, evaluation, prediction, and model interpretability.

The system combines:

- EfficientNetB0 with ImageNet pretrained weights
- TensorFlow and Keras for model development
- OpenCV for image processing
- FastAPI for model inference APIs
- Streamlit for the user interface
- Grad-CAM for model interpretability

## Key Features

- Four-class brain MRI classification
- EfficientNetB0-based transfer learning
- Image preprocessing and data augmentation
- Prediction confidence scoring
- Grad-CAM visual explanations
- Modular training and prediction pipelines
- FastAPI inference endpoints
- Streamlit-based interactive interface
- Comprehensive model evaluation
- Confusion matrix and per-class performance analysis
- Average inference time measurement

## System Architecture

```text
                         MRI Image
                             |
                             v
                    Image Preprocessing
                             |
                             v
                       EfficientNetB0
                             |
                             v
                     Softmax Prediction
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
          Prediction Pipeline         Grad-CAM
                 |                       |
                 v                       v
       Tumor Type + Confidence       Heatmap
                 |                       |
                 +-----------+-----------+
                             |
                             v
                     Streamlit Interface
```

## Model Architecture

The project uses EfficientNetB0 as the pretrained convolutional base with ImageNet weights.

A custom classification head is added on top of the base model:

```text
Input Image
    |
    v
224 x 224 x 3
    |
    v
EfficientNetB0
ImageNet Weights
    |
    v
Global Average Pooling
    |
    v
Dropout (0.4)
    |
    v
Dense Layer (256, ReLU)
    |
    v
Dropout (0.3)
    |
    v
Dense Layer
Softmax Activation
    |
    v
4 Output Classes
```

During the initial transfer learning stage, the EfficientNetB0 base model is frozen while the custom classification head is trained.

### Training Configuration

| Parameter | Value |
|---|---|
| Architecture | EfficientNetB0 |
| Pretrained Weights | ImageNet |
| Input Size | 224 × 224 × 3 |
| Dense Layer | 256 neurons |
| Output Classes | 4 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 32 |
| Maximum Epochs | 10 |
| Early Stopping Patience | 3 |

## Dataset

The dataset is organized into training and testing directories according to the four classification categories.

```text
artifacts/
└── raw/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    │
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

The project uses **7,200+ MRI scans** for model development and evaluates the trained model using **1,600 MRI images**.

The evaluation dataset contains 400 images per class:

| Class | Images |
|---|---:|
| Glioma | 400 |
| Meningioma | 400 |
| No Tumor | 400 |
| Pituitary | 400 |
| **Total** | **1,600** |

## Data Preprocessing

All input images are resized to **224 × 224 pixels** before being passed to EfficientNetB0.

EfficientNet-specific preprocessing is applied to the images.

Training images are additionally augmented using:

- Rotation range: 10 degrees
- Zoom range: 10%
- Width shift range: 10%
- Height shift range: 10%

The evaluation images are processed without augmentation.

## Training Pipeline

The training workflow is implemented as a modular pipeline:

```text
Data Ingestion
      |
      v
Data Transformation
      |
      v
Image Generators
      |
      v
EfficientNetB0
      |
      v
Model Training
      |
      v
Validation
      |
      v
Best Model Checkpoint
```

### Data Ingestion

The data ingestion component prepares the raw dataset for the subsequent transformation and training stages.

### Data Transformation

The data transformation component creates TensorFlow image generators for training and evaluation.

It performs:

- Image loading
- Image resizing
- EfficientNet preprocessing
- Training augmentation
- Class mapping
- Batch generation

### Model Training

The model trainer:

1. Initializes EfficientNetB0 with ImageNet weights.
2. Freezes the pretrained base model.
3. Adds a custom classification head.
4. Compiles the model using Adam and categorical cross-entropy.
5. Trains the model for up to 10 epochs.
6. Uses early stopping based on validation accuracy.
7. Saves the best-performing model checkpoint.

The best model is stored at:

```text
models/best_model.keras
```

## Model Evaluation

The trained model was evaluated on **1,600 MRI images**.

### Overall Performance

| Metric | Result |
|---|---:|
| Test Accuracy | **86.62%** |
| Macro Precision | **87.10%** |
| Macro Recall | **86.62%** |
| Macro F1-Score | **86.15%** |
| Weighted F1-Score | **86.15%** |
| Average Inference Time | **10.84 ms/image** |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Glioma | 93% | 67% | 78% | 400 |
| Meningioma | 80% | 81% | 81% | 400 |
| No Tumor | 91% | 99% | 95% | 400 |
| Pituitary | 84% | 99% | 91% | 400 |
| **Macro Average** | **87%** | **87%** | **86%** | **1,600** |

The model achieved particularly strong recall for the **No Tumor** and **Pituitary** classes. Glioma had the lowest recall at **67%**, making it an important area for further improvement.

### Confusion Matrix

```text
                 Predicted
              Glioma  Meningioma  No Tumor  Pituitary

Glioma           269       78          27        26
Meningioma        18      325          12        45
No Tumor           1        1         396         2
Pituitary          1        3           0       396
```

## Prediction Pipeline

The prediction pipeline loads the trained model and processes an input MRI image through the following stages:

```text
Input MRI
    |
    v
Load Image
    |
    v
Resize to 224 x 224
    |
    v
EfficientNet Preprocessing
    |
    v
Model Inference
    |
    v
Softmax Probabilities
    |
    v
Highest Probability Class
    |
    +----------------------+
    |          |           |
    v          v           v
Tumor      Tumor Type   Confidence
Detected
```

The prediction pipeline uses the following class mapping:

```text
0 -> glioma
1 -> meningioma
2 -> notumor
3 -> pituitary
```

The system determines whether a tumor is detected based on the predicted class. If the predicted class is `notumor`, `tumor_detected` is returned as `false`; otherwise, it is returned as `true`.

### Prediction Response

The prediction endpoint returns a response in the following format:

```json
{
    "tumor_detected": true,
    "tumor_type": "glioma",
    "confidence": 91.42
}
```

## Grad-CAM Explainability

The project integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize regions of the MRI image that influence the model's prediction.

Grad-CAM is generated using EfficientNetB0's `top_conv` layer.

The process is:

```text
MRI Image
    |
    v
EfficientNetB0
    |
    v
Predicted Class
    |
    v
Gradient Computation
    |
    v
Gradient Pooling
    |
    v
Activation Heatmap
    |
    v
Heatmap Resizing
    |
    v
Overlay with Original MRI
```

The implementation:

1. Preprocesses the MRI image.
2. Obtains the predicted class from the model.
3. Computes the gradients of the predicted class with respect to the `top_conv` activations.
4. Performs global average pooling on the gradients.
5. Generates the Grad-CAM heatmap.
6. Resizes the heatmap to the original image dimensions.
7. Applies a color map using OpenCV.
8. Overlays the heatmap on the original MRI.

This provides an interpretable visualization of the regions contributing to the model's classification.

## FastAPI Backend

The backend is implemented using **FastAPI** and provides endpoints for model prediction and Grad-CAM generation.

### `POST /predict`

Accepts an uploaded MRI image and returns the model prediction.

Example response:

```json
{
    "tumor_detected": true,
    "tumor_type": "meningioma",
    "confidence": 87.35
}
```

### `POST /gradcam`

Accepts an uploaded MRI image and generates a Grad-CAM visualization.

The endpoint returns the generated heatmap as an image.

## Streamlit Frontend

The project includes a **Streamlit** interface for interactive MRI analysis.

The interface allows users to:

1. Upload an MRI image.
2. View the original MRI.
3. Analyze the image.
4. View the predicted tumor type.
5. View the prediction confidence.
6. View the Grad-CAM heatmap.
7. Inspect the complete prediction response.

The interface displays three primary prediction metrics:

- Tumor Detected
- Tumor Type
- Confidence

## Application Workflow

```text
              Upload MRI Image
                     |
                     v
            Streamlit Interface
                     |
                     v
              FastAPI Backend
                     |
            +--------+--------+
            |                 |
            v                 v
       /predict           /gradcam
            |                 |
            v                 v
      Model Prediction     Grad-CAM
            |                 |
            v                 v
    Tumor Type +         Heatmap
     Confidence              |
            |                 |
            +--------+--------+
                     |
                     v
              Results Display
```

## Project Structure

The large MRI dataset inside `artifacts` is intentionally not expanded.

```text
Brain_Tumor_Detection_System/
│
├── artifacts/
│   └── raw/
│       ├── Training/
│       └── Testing/
│
├── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── routes.py
│   └── schemas.py
│
├── frontend/
│   ├── __init__.py
│   ├── requirements_frontend.txt
│   └── streamlit_app.py
│
├── models/
│   └── best_model.keras
│
├── screenshots/
│
├── src/
│   ├── __init__.py
│   ├── utils.py
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── gradcam.py
│   │   ├── model_evaluation.py
│   │   └── model_trainer.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── predict_pipeline.py
│       └── train_pipeline.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## Technologies Used

### Programming Language

- Python

### Machine Learning and Computer Vision

- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- Scikit-learn
- OpenCV

### Backend

- FastAPI
- Pydantic

### Frontend

- Streamlit
- Requests

### Development Tools

- Git
- GitHub

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/swagat994/Brain_Tumor_Detection_System.git
cd Brain_Tumor_Detection_System
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install Machine Learning and Backend Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Frontend Dependencies

```bash
pip install -r frontend/requirements_frontend.txt
```

## Dataset Setup

Place the MRI dataset in the following structure:

```text
artifacts/
└── raw/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    │
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

Ensure that the class directory names correspond to the expected class mapping:

```text
0 -> glioma
1 -> meningioma
2 -> notumor
3 -> pituitary
```

## Training the Model

From the project root, run:

```bash
python -m src.pipeline.train_pipeline
```

The training pipeline performs:

```text
Data Ingestion
      |
      v
Data Transformation
      |
      v
Data Generator Creation
      |
      v
EfficientNetB0 Initialization
      |
      v
Model Training
      |
      v
Checkpointing
      |
      v
Best Model
```

The best model is saved to:

```text
models/best_model.keras
```

## Evaluating the Model

Run the evaluation component:

```bash
python -m src.components.model_evaluation
```

The evaluation process reports:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Per-class precision
- Per-class recall
- Per-class F1-score
- Confusion matrix
- Average inference time

## Running the Backend

Start the FastAPI backend using the backend application entry point.

For example:

```bash
uvicorn backend.app:app --reload
```

The API will then be available through the configured local server.

## Running the Frontend

Start the Streamlit application:

```bash
streamlit run frontend/streamlit_app.py
```

The application provides a browser-based interface for uploading and analyzing MRI images.

## Example Usage

After starting the application:

1. Upload an MRI image in JPG, JPEG, or PNG format.
2. View the original MRI.
3. Select the analysis option.
4. The image is sent to the prediction API.
5. The model predicts one of the four tumor categories.
6. The prediction confidence is calculated.
7. Grad-CAM generates a visualization of influential image regions.
8. The original MRI and heatmap are displayed.
9. Prediction details are shown in the interface.

## Limitations

- Model performance varies across tumor categories.
- **Glioma recall is 67%**, which is lower than the recall achieved for the other classes.
- The reported evaluation results are based on the project's current dataset and evaluation procedure.
- Grad-CAM provides an interpretation of model behavior but does not establish clinical validity.
- Model performance may vary on MRI scans from different datasets, scanners, imaging protocols, or clinical environments.
- The system is not clinically validated and should not be used as a substitute for professional medical diagnosis.

## Future Improvements

Potential improvements include:

- Fine-tuning additional EfficientNetB0 layers.
- Improving recall for the Glioma class.
- Performing systematic hyperparameter optimization.
- Experimenting with alternative CNN architectures.
- Evaluating the model on independent external datasets.
- Investigating class balancing techniques.
- Exploring additional model explainability techniques.
- Improving robustness across different MRI acquisition conditions.
- Expanding the evaluation dataset to assess model generalization.

## License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

## Disclaimer

This project is developed for **educational and research purposes only**.

It is not intended for clinical diagnosis, treatment decisions, or other medical decision-making. Predictions generated by this system should not be considered a substitute for evaluation by a qualified medical professional.
