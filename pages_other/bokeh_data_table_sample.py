# import streamlit as st
# from bokeh.plotting import figure
# from bokeh.models import ColumnDataSource, CustomJS
# from bokeh.models import DataTable, TableColumn, HTMLTemplateFormatter, DateFormatter
# import pandas as pd
# df_source = pd.DataFrame({"tweet": ['test', 'test2'], "favs": [0,4]})


# def show_bokeh_table_not_working():
#     # この show_bokeh_table_not_working() は動かない。
#     # この関数は次のエラーが javascript コンソールにでてしま
#     # Uncaught (in promise) Error: Model 'DataTable' does not exist. This could be due to a widget or a custom model not being registered before first usage.

#     cds_full = ColumnDataSource(df_source)
#     columns_full = [
#         TableColumn(field="favs", title='Reply @ Twitter', width=85),
#         TableColumn(field='tweet', title='Tweet', width = 100),
#     ]

#     cds_full.selected.js_on_change("indices",
#                                 CustomJS(args=dict(source=cds_full),
#                                             code="""document.dispatchEvent(
#                                             new CustomEvent("INDEX_SELECT", {detail: {data: source.selected.indices}}))"""))

#     p_full = DataTable(source=cds_full, columns=columns_full, css_classes=["my_table"], row_height=150,
#                     editable=False, reorderable=True, scroll_to_selection=True, sortable=True,
#                     selectable="checkbox", # to enable checkbox in the datatable, next to the index col
#                     aspect_ratio='auto', width=770)

#     st.bokeh_chart(p_full)

# # show_bokeh_table_not_working() は動かないので、次のように書き換える
# def show_bokeh_table_working():
#     cds_full = ColumnDataSource(df_source)
#     columns_full = [
#         TableColumn(field="favs", title='Reply @ Twitter', width=85),
#         TableColumn(field='tweet', title='Tweet', width = 100),
#     ]

#     cds_full.selected.js_on_change("indices",
#                                 CustomJS(args=dict(source=cds_full),
#                                             code="""document.dispatchEvent(
#                                             new CustomEvent("INDEX_SELECT", {detail: {data: source.selected.indices}}))"""))

#     p_full = DataTable(source=cds_full, columns=columns_full, css_classes=["my_table"], row_height=150,
#                     editable=False, reorderable=True, scroll_to_selection=True, sortable=True,
#                     selectable="checkbox", # to enable checkbox in the datatable, next to the index col
#                     aspect_ratio='auto', width=770)

#     # st.bokeh_chart(p_full)
#     st.bokeh_chart(p_full.to_json(include_defaults=True))

# show_bokeh_table_working()