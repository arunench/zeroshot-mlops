from __future__ import annotations


def main() -> None:
    from app.utils.config import ConfigError, load_settings

    try:
        settings = load_settings()
    except ConfigError as exc:
        import streamlit as st

        st.set_page_config(page_title="Zero-Shot Classification")
        st.title("Zero-Shot Text Classification")
        st.error(f"Configuration error: {exc}")
        st.stop()

    from app.utils.logging_config import configure_logging

    configure_logging()
    import streamlit as st

    st.set_page_config(page_title="Zero-Shot Classification")
    st.title("Zero-Shot Text Classification")
    st.write(
        "A local MLOps learning platform for managing zero-shot models, "
        "candidate labels, and persisted classifications."
    )
    st.caption(f"Application version: {settings.app_version}")
    st.info("Use the page navigation to manage configured models.")


if __name__ == "__main__":
    main()
