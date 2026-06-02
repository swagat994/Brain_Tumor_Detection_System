import requests
import streamlit as st


st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MRI Brain Tumor Detection System")

st.markdown(
    "Upload an MRI scan to detect the presence of a brain tumor and visualize the affected region using Grad-CAM."
)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original MRI")

        st.image(
            uploaded_file,
            use_container_width=True
        )

    if st.button("Analyze MRI"):

        with st.spinner("Analyzing MRI..."):

            # -------------------------
            # Prediction Request
            # -------------------------

            uploaded_file.seek(0)

            prediction_response = requests.post(
                "https://brain-tumor-detection-system-1-73ft.onrender.com/predict",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }
            )

            prediction_result = prediction_response.json()

            # -------------------------
            # GradCAM Request
            # -------------------------

            uploaded_file.seek(0)

            gradcam_response = requests.post(
                "https://brain-tumor-detection-system-1-73ft.onrender.com/gradcam",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }
            )

            gradcam_path = "temp_gradcam.jpg"

            with open(
                gradcam_path,
                "wb"
            ) as f:

                f.write(
                    gradcam_response.content
                )

        with col2:

            st.subheader("Grad-CAM Heatmap")

            st.image(
                gradcam_path,
                use_container_width=True
            )

        st.success("Analysis Complete")

        st.markdown("---")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:

            st.metric(
                "Tumor Detected",
                str(
                    prediction_result[
                        "tumor_detected"
                    ]
                )
            )

        with metric_col2:

            st.metric(
                "Tumor Type",
                prediction_result[
                    "tumor_type"
                ]
            )

        with metric_col3:

            st.metric(
                "Confidence",
                f"{prediction_result['confidence']}%"
            )

        st.markdown("---")

        st.subheader("Prediction Details")

        st.json(
            prediction_result
        )