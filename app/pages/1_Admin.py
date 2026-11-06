from __future__ import annotations

import streamlit as st

from app.services.model_manager import ModelManager, ModelManagerError
from app.utils.config import ConfigError, load_settings

st.set_page_config(page_title="Administrator", layout="wide")
st.title("Administrator")
st.caption("Manage configured models and the local Hugging Face cache.")

try:
    settings = load_settings()
    manager = ModelManager(settings)
except ConfigError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

statuses = manager.list_models()
st.subheader("Configured models")
models_by_key = {model.key: model for model in settings.models}
for status in statuses:
    model = models_by_key[status.model_key]
    with st.expander(f"{model.name} - {status.state}", expanded=status.state != "ready"):
        st.text(f"Identifier: {model.model_id}")
        st.text(f"Revision: {status.revision or 'unpinned'}")
        st.text(f"Local directory: {status.local_dir or 'Unavailable'}")
        st.text(f"Hugging Face cache: {status.cache_location or 'Unavailable'}")
        if status.message:
            st.caption(status.message)

model_choices = {model.key: f"{model.name} ({model.model_id})" for model in settings.models}
selected_keys = st.multiselect(
    "Models to download",
    options=list(model_choices),
    format_func=lambda key: model_choices[key],
    key="admin_download_models",
)
if st.button("Download selected models", disabled=not selected_keys):
    with st.spinner("Downloading and validating selected model snapshots..."):
        for model_key in selected_keys:
            result = manager.download_model(model_key)
            if result.state == "ready":
                st.success(f"{model_choices[model_key]} is ready.")
            else:
                st.error(result.message or f"{model_choices[model_key]} could not be prepared.")
    st.rerun()

left, right = st.columns(2)
with left:
    st.subheader("Delete a downloaded model")
    downloaded = [
        status.model_key for status in statuses if status.local_dir and status.local_dir.exists()
    ]
    delete_key = st.selectbox(
        "Model",
        options=downloaded,
        format_func=lambda key: model_choices[key],
        index=None,
        placeholder="Select a local model",
        key="admin_delete_model",
    )
    confirmed_delete = st.checkbox(
        "I understand this permanently removes the selected managed model.",
        key="admin_confirm_model_delete",
        disabled=delete_key is None,
    )
    if st.button(
        "Delete selected model",
        disabled=delete_key is None or not confirmed_delete,
        type="secondary",
    ):
        try:
            manager.delete_model(delete_key)
        except (ModelManagerError, OSError) as exc:
            st.error(f"Model deletion failed: {exc}")
        else:
            st.success(f"{model_choices[delete_key]} was deleted.")
            st.rerun()
    if not downloaded:
        st.info("No managed model snapshots are present.")

with right:
    st.subheader("Clear Hugging Face cache")
    confirmed_cache = st.checkbox(
        "I understand this removes the local Hugging Face Hub cache.",
        key="admin_confirm_cache_clear",
    )
    if st.button("Clear Hugging Face cache", disabled=not confirmed_cache):
        try:
            st.info(manager.clear_huggingface_cache())
        except ModelManagerError as exc:
            st.error(f"Cache cleanup failed: {exc}")

if st.button("Refresh model list"):
    manager.refresh_models()
    st.rerun()
