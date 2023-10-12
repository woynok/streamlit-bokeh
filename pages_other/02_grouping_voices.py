import numpy as np
import pickle
from matplotlib import pyplot as plt
import streamlit as st
from streamlit.components.v1 import html
from wordcloud import WordCloud
from lib.text_data.text_data import TextData
from lib.analysis import DocumentMap, WordStatistics
from lib.visualize.word_clouds_scatter import WordCloudsScatter

st.set_page_config(layout="wide")

@st.cache_data
def get_data()->tuple[TextData, DocumentMap, WordStatistics]:
    td = TextData.load("data/app_reviews_thums_up.pkl")
    with open("data/documentmap_thums_up.pkl", "rb") as f:
        dm = pickle.load(f)
    return td, dm

td, dm = get_data()
ws = WordStatistics(td.docs, min_document_frequency = 0.02, max_document_frequency=1.00)

st.header("1. 声のグループ分け")
st.markdown("2～8のグループに機械的に分けることができます。名前のつけやすい数を見つけてグループに名前をつけましょう")

n_clusters = st.slider("分けるグループの数", min_value=2, max_value=14, value = 4, step = 1)
word_cloud_mode = st.toggle("ワードクラウドを表示する", value = True)

tfidf = ws.calculate_tf_idf(labels = dm.dict_labels[n_clusters], n_words = 200)

if word_cloud_mode:
    wcs = WordCloudsScatter(
        labels_unique=tfidf.labels_unique,
        d_vectors=tfidf.d_vectors,
        xs=dm.xs,
        ys=dm.ys,
        labels=dm.dict_labels[n_clusters],
        array_size = 600,
    )
    # for wc in wcs:
    #     st.image(wc.to_image(), width=300)
    
    figure = plt.figure()
    ax = figure.gca()
    for ilabel, wc in enumerate(wcs):
        array = wc.to_array()
        x = wcs.positions.xs_cloud[ilabel]
        y = wcs.positions.ys_cloud[ilabel]
        width = wcs.positions.sizes_cloud[ilabel]
        ax.imshow(array, extent=(x - width, x + width, y - width, y + width))
        ax.text(x + wcs.positions.sizes_cloud[ilabel]*0.8, y + wcs.positions.sizes_cloud[ilabel]*0.8, str(wcs.labels_unique[ilabel]), ha="center", va="center", fontsize=5)
    ax.set_xlim(wcs.xlim)
    ax.set_ylim(wcs.ylim)
    # st.pyplot(figure)
    # st.write(array.shape)


import numpy as np

from bokeh.plotting import figure, show
from bokeh.models import ImageRGBA, ColumnDataSource

N = 20
img = np.empty((N,N), dtype=np.uint32)
view = img.view(dtype=np.uint8).reshape((N, N, 4))
for i in range(N):
    for j in range(N):
        view[i, j, 0] = int(i/N*255)
        view[i, j, 1] = 158
        view[i, j, 2] = int(j/N*255)
        view[i, j, 3] = 255

st.write(img.shape)
st.write(img)


p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
p.x_range.range_padding = p.y_range.range_padding = 0

im_glyph = ImageRGBA(x=0, y=0, dw=1, dh=1)

im_source = ColumnDataSource(dict(
    image = [img]
))
p.add_glyph(im_source,im_glyph)
st.bokeh_chart(p,)
