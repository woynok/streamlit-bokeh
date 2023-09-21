import pandas as pd
import streamlit as st
from lib.text_data.text_data import TextData
from lib.analysis import DocumentMap, WordStatistics
import pickle

if st.button("read csv data and perform heavy processing"):
    df = pd.read_csv("data/app_reviews_thums_up.csv")
    td = TextData.from_dataframe(df, text_column_name="content")
    td.calculate_embeddings()
    td.tokenize_docs()
    td.save("data/app_reviews_thums_up.pkl")

td = TextData.load("data/app_reviews_thums_up.pkl")

if st.toggle("load document map from pickle", value = True):
    with open("data/documentmap_thums_up.pkl", "rb") as f:
        dm = pickle.load(f)
else:
    with st.spinner("document map..."):
        dm = DocumentMap(td.embeddings)

with st.spinner("word statistics..."):
    ws = WordStatistics(td.docs, min_document_frequency = 0.01, max_document_frequency=1.00)

if st.button("update document map"):
    with open("data/documentmap_thums_up.pkl", "wb") as f:
        pickle.dump(dm, f)

