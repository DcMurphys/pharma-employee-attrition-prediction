"""Download button for prepared CSV bytes (the disclaimer is already in the file)."""

import streamlit as st


def csv_download(label: str, data: bytes, file_name: str, *, key: str) -> None:
    """Secondary download button for a CSV file."""
    st.download_button(
        label,
        data,
        file_name=file_name,
        mime="text/csv",
        icon=":material/download:",
        key=key,
        on_click="ignore",
    )
