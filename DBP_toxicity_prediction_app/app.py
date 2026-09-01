"""Combined entry point for prediction and public usage statistics."""

import streamlit as st

from analytics import log_visit_once


st.set_page_config(
    page_title="DBP Toxicity Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Count a browser session once regardless of which page the visitor opens first.
log_visit_once()

pages = [
    st.Page(
        "prediction_page.py",
        title="Toxicity Predictor",
        icon=":material/science:",
        default=True,
    ),
    st.Page(
        "usage_statistics_page.py",
        title="Usage Statistics",
        icon=":material/monitoring:",
    ),
]

navigation = st.navigation(pages, position="top")
navigation.run()
