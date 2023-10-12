import numpy as np
import pickle
from matplotlib import pyplot as plt
import streamlit as st
from streamlit.components.v1 import html
from wordcloud import WordCloud
from bokeh.models import Column
from bokeh.plotting import save
from lib.text_data.text_data import TextData
from lib.analysis import DocumentMap, WordStatistics
from lib.visualize.word_clouds_scatter import WordCloudsScatter
from lib.visualize.docs_scatter import DocumentVisualizer

st.set_page_config(layout="wide")

@st.cache_data
def get_data()->tuple[TextData, DocumentMap, WordStatistics]:
    td = TextData.load("data/app_reviews_thums_up.pkl")
    with open("data/documentmap_thums_up.pkl", "rb") as f:
        dm = pickle.load(f)
    return td, dm

@st.cache_data
def get_wcs(
    _docs,
    n_clusters, n_words,
    min_document_frequency, max_document_frequency
    )->WordCloudsScatter:

    ws = WordStatistics(docs=_docs, min_document_frequency = min_document_frequency, max_document_frequency=max_document_frequency)
    tfidf = ws.calculate_tf_idf(labels = dm.dict_labels[n_clusters], n_words = n_words)
    wcs = WordCloudsScatter(
        labels_unique=tfidf.labels_unique,
        d_vectors=tfidf.d_vectors,
        xs=dm.xs,
        ys=dm.ys,
        labels=dm.dict_labels[n_clusters],
        array_size = 900,
    )
    return wcs

td, dm = get_data()

st.header("1. 声のグループ分け")
st.markdown(f"""
            AIでユーザーの声を似ているものにグループ分けして、色分けして表現しています。

            各グループに名前をつけてください。
            """)

n_clusters = st.slider("グループの数", min_value=2, max_value=14, value = 7, step = 1)
word_cloud_mode = st.toggle("ワードクラウドでグループの特徴を表示する", value = False)
d_labels_wamei = st.session_state.get(f"d_labels_wamei_{n_clusters}", dict())
wcs = get_wcs(_docs = td.docs, n_clusters = n_clusters, n_words = 300, min_document_frequency = 0.02, max_document_frequency=1.00)

# aria-label という attribute が "0の名前" となっている input エレメントを直接もつdivエレメントのbackground color を red　にする css

cmap = wcs.get_cmap(n_clusters)
css_texts = [f"""div.stTextInput:has(input[aria-label="{label}の名前"]) p escaped_open color: {cmap[ilabel]}; escaped_close""".replace("escaped_open", "{").replace("escaped_close", "}") for ilabel, label in enumerate(wcs.labels_unique)]
css_text = "<style>\n" + "\n".join(css_texts) + "\n</style>"

st.markdown(css_text, unsafe_allow_html=True)
with st.expander("グループに名前をつける", expanded = True):
    nlabels = len(wcs.labels_unique)
    # 8列で表示する
    ncols = 10
    nrows = int(np.ceil(nlabels / ncols))
    ilabel = 0
    for ilabel, label in enumerate(wcs.labels_unique):
        if ilabel % ncols == 0:
            cols = st.columns([1/ncols] * ncols)
        with cols[ilabel % ncols]:
            d_labels_wamei[label] = st.text_input(
                f"{label}の名前",
                value = d_labels_wamei.get(label, ""),
                key = f"{ilabel}_wamei",
                )
    if ilabel % ncols != ncols - 1:
        # 最後の列が埋まっていない場合は空白で埋める
        for _ in range(ncols - ilabel % ncols - 1):
            cols[ilabel % ncols].empty()
    st.session_state[f"d_labels_wamei_{n_clusters}"] = d_labels_wamei

dv = DocumentVisualizer(td.docs)
if word_cloud_mode:
    d_labels_wamei = st.session_state.get(f"d_labels_wamei_{n_clusters}", dict())
    d_labels_wamei = {k:v for k,v in d_labels_wamei.items() if v != "" and v != k}
    wcs.d_labels_wamei = d_labels_wamei
    # st.write(wcs.to_dataframe())
    with st.expander("詳細設定"):
        size_factor = st.slider("ワードクラウドの大きさ(クラウド同士がかぶっていて見えないときにお使いください)", min_value=1.0, max_value=10.0, value = 5.0, step = 0.1)
        alpha_factor = st.slider("ワードクラウドの透明度", min_value=0.0, max_value=1.0, value = 0.5, step = 0.01)
        power_fill_alpha = np.log(0.07) / np.log(0.5)
        wcs.fill_alpha = alpha_factor ** (power_fill_alpha)
        power_line_alpha = np.log(0.65) / np.log(0.5)
        wcs.line_alpha = alpha_factor ** (power_line_alpha)
    scatter_result = wcs.to_bokeh(
        size_factor=size_factor / 5 * 0.1,
        canvas_size=(1000,500)
        )
    link_by = "label"
    d_labels_to_text = dv.get_d_labels_to_texts(dm.dict_labels[n_clusters])
else:
    d_labels_wamei = st.session_state.get(f"d_labels_wamei_{n_clusters}", dict())
    d_labels_wamei = {k:v for k,v in d_labels_wamei.items() if v != "" and v != k}
    scatter_result = dv.get_scatter_from_dm(
        dm, n_clusters = 7, d_labels_wamei = d_labels_wamei,
        update=True,
    )
    link_by = "indices"
    d_labels_to_text = None

table_result = dv.get_table_from_dm(
    dm, n_clusters = 7, d_labels_wamei = d_labels_wamei, show_empty_table=True,
    update=True,
    )

DocumentVisualizer.link_to_table(
    scatter_result, table_result, link_by = link_by,
    d_labels_to_texts=d_labels_to_text,
)
bokeh_result = Column(scatter_result.scatter, table_result.table)

# streamlit 1.24 以降から bokeh のDataTable が動かなくなったのでhtml表示
# st.bokeh_chart(bokeh_result)
html_str = bokeh_result._repr_html_()
# # save(bokeh_result, "tmp.html")
# with open("tmp.html") as f:
#     html_str = f.read()
html(html_str, width=1000, height=700)

