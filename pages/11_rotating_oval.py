import streamlit as st

@st.cache_data(ttl = 10)
def get_image():
    with open("rotating_oval.svg", "r") as f:
        svg_string = f.read()
    return svg_string

with open("rotating_oval.svg", "r") as f:
    svg_string = f.read()

button = st.button("start")
plceholder = st.empty()
plceholder.image("rotating_oval.svg")

