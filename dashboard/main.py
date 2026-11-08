from pathlib import Path

import streamlit as st


DASHBOARD_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="DefenseAI",
    layout="wide",
)

required_files = [
    DASHBOARD_DIR / "app.py",
    DASHBOARD_DIR / "predict.py",
    DASHBOARD_DIR / "documents.py",
]

missing = [
    path.name
    for path in required_files
    if not path.exists()
]

if missing:
    st.error("Required application files are missing.")
    st.write(missing)
    st.stop()

pages = [
    st.Page(
        "app.py",
        title="Validation Dashboard",
        url_path="validation",
        default=True,
    ),
    st.Page(
        "predict.py",
        title="Sensor Inference",
        url_path="inference",
    ),
    st.Page(
        "documents.py",
        title="Document Search",
        url_path="documents",
    ),
]

selected_page = st.navigation(
    pages,
    position="sidebar",
)

selected_page.run()