from __future__ import annotations

import streamlit as st

from app.services.classifier import ClassificationError, Classifier
from app.services.minio_storage import MinIOStorage, StorageError
from app.services.model_manager import ModelManager
from app.utils.config import ConfigError, load_settings

st.set_page_config(page_title="Classify Text", layout="wide")
st.title("Classify text")
st.caption("Choose a ready model and a saved candidate-label set.")


@st.cache_resource
def load_services() -> tuple[object, ModelManager, Classifier, MinIOStorage]:
    loaded_settings = load_settings()
    model_manager = ModelManager(loaded_settings)
    return (
        loaded_settings,
        model_manager,
        Classifier(loaded_settings, model_manager),
        MinIOStorage(loaded_settings),
    )


try:
    settings, model_manager, classifier, storage = load_services()
except ConfigError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

ready_models = [
    model for model in settings.models if model_manager.status(model.key).state == "ready"
]
if not ready_models:
    st.error("No model is downloaded and ready. Ask the administrator to download a model.")
    st.stop()
if not settings.label_sets:
    st.error("No saved candidate-label sets are configured. Ask the administrator to add one.")
    st.stop()

model_labels = {model.key: f"{model.name} ({model.model_id})" for model in ready_models}
label_set_labels = {label_set.key: label_set.name for label_set in settings.label_sets}
default_set_index = next(
    (
        index
        for index, label_set in enumerate(settings.label_sets)
        if label_set.key == settings.default_set
    ),
    0,
)

with st.form("classification_form"):
    model_key = st.selectbox(
        "Model",
        options=list(model_labels),
        format_func=lambda key: model_labels[key],
    )
    label_set_key = st.selectbox(
        "Candidate-label set",
        options=list(label_set_labels),
        index=default_set_index,
        format_func=lambda key: label_set_labels[key],
    )
    input_text = st.text_area(
        "Text to classify",
        max_chars=10_000,
        height=180,
        help="Enter up to 10,000 characters.",
    )
    submitted = st.form_submit_button("Classify")

if submitted:
    selected_label_set = next(
        label_set for label_set in settings.label_sets if label_set.key == label_set_key
    )
    try:
        prediction = classifier.classify(
            input_text,
            model_key,
            selected_label_set.labels,
        )
        storage.write_prediction(prediction)
    except ClassificationError as exc:
        st.error(str(exc))
    except StorageError:
        st.error("Unable to save this classification to MinIO. No prediction was recorded.")
    else:
        st.success("Classification completed and saved to MinIO.")
        st.subheader("Result")
        st.text(f"Predicted class: {prediction['predicted_class']}")
        st.text(f"Confidence: {prediction['confidence']:.4f}")
        st.text(f"Model: {prediction['model_name']} ({prediction['model_id']})")
        st.text(f"Timestamp: {prediction['timestamp']}")
        st.text("Ranked candidate labels:")
        for item in prediction["ranked_predictions"]:
            st.text(f"  {item['label']}: {item['score']:.4f}")
