import streamlit as st
from streamlit.components.v1 import html

@st.cache_data
def show_pyvis():
    with open('data/gameofthrones.html', 'r') as f:
        file_content = f.read()
    html(file_content, width=800, height=800, scrolling=True)

show_pyvis()