"""Thin Streamlit entry point: `python -m streamlit run app.py`."""

import streamlit as st

from procurement import config

config.ensure_env_file()
config.load_env()

from procurement.llm import LLMError, client_from_env  # noqa: E402
from procurement.store import Store  # noqa: E402
from procurement.ui import render_workspace  # noqa: E402

st.set_page_config(page_title="Procurement evidence workspace", layout="wide")

# Building the client makes no request; requests happen only in explicit button handlers.
if "client" not in st.session_state:
    try:
        st.session_state.client = client_from_env()
    except LLMError as exc:
        st.session_state.client = None
        st.error(str(exc))

render_workspace(Store(config.DEFAULT_DB_PATH), st.session_state.client)
