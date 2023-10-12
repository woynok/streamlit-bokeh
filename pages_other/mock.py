import pandas as pd
import streamlit as st
from lib.text_data.text_data import TextData
from lib.analysis import DocumentMap, WordStatistics
import pickle

import pandas as pd
if st.button("read csv data and perform heavy processing"):
    df = pd.read_csv("data/app_reviews_thums_up.csv")
    td = TextData.from_dataframe(df, text_column_name="content")
    td.calculate_embeddings()
    td.tokenize_docs()
    td.save("data/app_reviews_thums_up.pkl")

td = TextData.load("data/app_reviews_thums_up.pkl")

with st.spinner("document map..."):
    dm = DocumentMap(td.embeddings)

with st.spinner("word statistics..."):
    ws = WordStatistics(td.docs, min_document_frequency = 0.01, max_document_frequency=1.00)

if st.button("save"):
    with open("data/wordstatistics_thums_up.pkl", "wb") as f:
        pickle.dump(ws, f)
    import pickle
    with open("data/documentmap_thums_up.pkl", "wb") as f:
        pickle.dump(dm, f)
