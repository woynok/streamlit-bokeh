import streamlit as st
import pickle
from bokeh.plotting import figure, ColumnDataSource, output_file, show, save
from bokeh.models import HoverTool, CustomJS, Slider, Button, Label, CategoricalColorMapper, Legend
from bokeh.models import DataSource, Range1d, LabelSet, Label, ColumnDataSource, Arrow, NormalHead, OpenHead, VeeHead
from bokeh.models import BoxSelectTool, LassoSelectTool, WheelZoomTool, PanTool, ResetTool, SaveTool, TapTool
# from bokeh.models import BoxZoomTool, ZoomInTool, ZoomOutTool, UndoTool, RedoTool, CrosshairTool, HoverTool
# from bokeh.models import CustomJS, Slider, Button, Label, CategoricalColorMapper, Legend, LegendItem
# from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label, ColumnDataSource, Arrow, NormalHead, OpenHead, VeeHead
from bokeh.models import DataTable, DataSource, TableColumn, Scatter

from bokeh.layouts import column, row
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap

@st.cache_data(ttl=60*5)
def get_data():
    with open('./data/result.pkl', 'rb') as f:
        tsne_two_dim, label, sentences = pickle.load(f)
    return tsne_two_dim, label, sentences

tsne_two_dim, label, sentences = get_data()

x = tsne_two_dim[:, 0]
y = tsne_two_dim[:, 1]
label = [str(i) for i in label]
# st.write(x.shape, y.shape, len(label), len(sentences))
hover = HoverTool(tooltips=[
    ("index", "$index"),
    # ("(x,y)", "($x, $y)"),
    # ("label", "@label"),
    # ("sentences", "@sentences"),
    ("review", "@sentences")
])

# cmap = factor_cmap('label', palette=Category20[20], factors=label)
label_unique = sorted(list(set(label)))
color = [Category10[10][label_unique.index(l)] for l in label]

source = ColumnDataSource(data=dict(x=x, y=y, label=label, sentences=sentences, color = color))

plot = figure(plot_width=800, plot_height=800, tools=[hover, BoxSelectTool(), LassoSelectTool(), WheelZoomTool(), ResetTool(), SaveTool(), TapTool()], title="Sentence Embedding Visualization")
plot.scatter('x', 'y', size=10, source=source, alpha=0.8, legend_group = 'label', color="color")
plot.legend.location = "top_left"
# plot.legend.click_policy="hide"

st.bokeh_chart(
    plot,
    use_container_width=True
)
# st.title('Sentence Embedding Visualization')
# st.write('This is a demo of visualizing sentence embedding using t-SNE.')
idx = st.text_input("input idx", value=0)
st.write(sentences[int(idx)])
