import streamlit as st
# from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, save

from bokeh.models import CustomJS, Slider, Label, CategoricalColorMapper, Legend, LegendItem
from bokeh.models.widgets import Button
from bokeh.layouts import Row, column

# from bokeh.transform import dodge

# output_file("dodged_bars.html")

words = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
years = ['全体の単語頻度', '選択した文書の単語頻度']

data = {'words' : words,
        '全体の単語頻度'   : [2, 1, 4, 3, 2, 4],
        '選択した文書の単語頻度'   : [5, 3, 3, 2, 4, 6],
}

source = ColumnDataSource(data=data)

p = figure(x_range=(0, 10), y_range=words, 
           height=500, title="word counts by year",
           toolbar_location=None, tools = "hover")

from bokeh.palettes import Category10
primary_color = Category10[10][7]
secondary_color = Category10[10][3]
p.hbar(y='words', right='全体の単語頻度', left = 0, source=source,
       color=primary_color, legend_label="全体の単語頻度", height = 0.8)

p.hbar(y='words', right='選択した文書の単語頻度', left = 0, source=source,
       color=secondary_color, legend_label="選択した文書の単語頻度", alpha = 0.8, height = 0.6)

# p の y 軸を、['Apples', 'Pears', 'Nectarines'] のみにする
# p.y_range.factors = ['Apples', 'Pears', 'Nectarines']

# bar chart に sort するボタンを作る
callback = CustomJS(args=dict(source=source, plot = p), code=open("a.js", encoding="utf-8").read())
button = Button(label="Sort", button_type="success")
button.js_on_event("button_click", callback)
from bokeh.layouts import gridplot
grid = gridplot([[p, button]])
# st.bokeh_chart(grid)


# # p.y_range.range_padding = 0.1
# # p.xgrid.grid_line_color = None
# # p.legend.location = "top_left"
# # p.legend.orientation = "horizontal"
# p.legend.click_policy="hide"

save(grid, filename="bokeh_word_bar.html", title="word counts by year")

from streamlit.components.v1 import html
html(open("bokeh_word_bar.html", 'r').read(), width=800, height=800)

# st.bokeh_chart(
#     Row(p, button),
# )
