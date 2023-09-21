from typing import Literal
import numpy as np
import pandas as pd

from bokeh.plotting import figure, save, show
from bokeh.models import BoxZoomTool, WheelZoomTool, PanTool, ResetTool, ColumnDataSource, Text, CustomJS, WheelPanTool, TapTool
from bokeh.models import Range1d, HoverTool, DataTable, TableColumn
from bokeh.plotting.figure import Figure
from lib.simple_docs import SimpleDocs
from lib.analysis import DocumentMap
from bokeh.palettes import Category10, Category20

def _get_labels_wamei(labels: np.ndarray, d_labels_wamei:dict[int, str])->list[str]:
    if d_labels_wamei is None:
        return labels
    else:
        return np.array([f"{label}:{d_labels_wamei.get(label, '')}" for label in labels])

default_empty_texts = ["点を選択するとここに文書が表示されます"] * 3
default_empty_labels = ["-"] * 3

class DocumentScatterResult:
    def __init__(
        self,
        source: ColumnDataSource,
        scatter: Figure,
    ):
        self.source: ColumnDataSource = source
        self.scatter: Figure = scatter

class DocumentScatterGenerator:
    def __init__(
        self,
        labels: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        texts: list[str],
        nlabels: int = None,
        colors: list[str] = None,
        d_labels_wamei: dict[int, str] = None,
    ):
        self.xs: np.ndarray = xs
        self.ys: np.ndarray = ys
        self.texts: list[str] = texts
        self.labels: np.ndarray = labels
        self.d_labels_wamei: np.ndarray = d_labels_wamei
        self.nlabels = nlabels if nlabels is not None else len(np.unique(labels))
        self.cmap = Category10[self.nlabels] if self.nlabels <= 10 else Category20[self.nlabels]
        self.colors = [self.cmap[label] for label in self.labels] if colors is None else colors

    @property
    def labels_wamei(self)->list[str]:
        return _get_labels_wamei(self.labels, self.d_labels_wamei)

    @property
    def result(self) -> DocumentScatterResult:
        source = ColumnDataSource(data = dict(
            x = self.xs,
            y = self.ys,
            text = self.texts,
            label = self.labels,
            color = self.colors,
            label_wamei = self.labels_wamei,
        ))
        hover = HoverTool(
            tooltips = [
                ("", "グループ:@label_wamei"),
                ("声", "@text"),
            ]
        )
        x_range = Range1d(
            start = self.xs.min() - 0.1,
            end = self.xs.max() + 0.1,
            bounds = (self.xs.min() - 0.1, self.xs.max() + 0.1),
        )
        y_range = Range1d(
            start = self.ys.min() - 0.1,
            end = self.ys.max() + 0.1,
            bounds = (self.ys.min() - 0.1, self.ys.max() + 0.1),
        )
        tools = ["lasso_select", "box_select", hover, "reset"]
        scatter = figure(
            title = "ユーザーの声を点として色分けして表示（選択すると本文が下の表に表示されます）",
            tools=tools,
            match_aspect=True,
            plot_width = 1000,
            plot_height = 500,
            x_range = x_range,
            y_range = y_range,
        )
        scatter.circle(
            'x', 'y',
            size = 10, source = source, 
            alpha = 0.8,
            color = 'color',
            legend_group = "label_wamei",
            )
        # scatter.legend.visible = False
        scatter.toolbar.logo = None
        scatter.toolbar_location = None
        # ticks を消す
        scatter.xaxis.visible = False
        scatter.yaxis.visible = False

        # grid を消す
        scatter.xgrid.grid_line_color = None
        scatter.ygrid.grid_line_color = None

        return DocumentScatterResult(source, scatter)

class DocumentTableResult:
    def __init__(
        self,
        source: ColumnDataSource,
        table: DataTable,
        empty_labels: list[int],
        empty_texts: list[str],
    ):
        self.source: ColumnDataSource = source
        self.table: DataTable = table
        self.empty_labels: list[int] = empty_labels
        self.empty_texts: list[str] = empty_texts

class DocumentTableGenerator:
    def __init__(
        self,
        labels: np.ndarray,
        texts: list[str],
        d_labels_wamei: dict[int, str] = None,
        show_empty_table = False,
        empty_labels: list[int] = None,
        empty_texts: list[str] = None,
    ):
        self.labels: np.ndarray = labels
        self.texts: list[str] = texts
        self.labels_wamei: list[str] = _get_labels_wamei(self.labels, d_labels_wamei)
        self.show_empty_table = show_empty_table
        self.empty_labels: list[int] = empty_labels
        self.empty_texts: list[str] = empty_texts
        
    @property
    def result(
            self,
        )->DocumentTableResult:
        
        bokeh_table_colmns = [
            TableColumn(field = "label_wamei", title = "グループ", width = 80),
            TableColumn(field = "text", title = "テキスト", width = 900),
        ]
        if self.show_empty_table:            
            source = ColumnDataSource(data = dict(
                label = self.empty_labels,
                label_wamei = self.empty_labels,
                text = self.empty_texts,
            ))
        else:
            source = ColumnDataSource(data = dict(
                label = self.labels,
                label_wamei = self.labels_wamei,
                text = self.texts,
            ))
        table = DataTable(
            source = source,
            columns = bokeh_table_colmns,
            width = 1000,
            height = 200,
            index_position = None,
            selectable = True,
            fit_columns = False,
            reorderable = False,
        )
        return DocumentTableResult(
            source = source,
            table = table,
            empty_labels = self.empty_labels,
            empty_texts = self.empty_texts,
        )


class DocumentVisualizer:
    def __init__(
        self,
        docs: list[str]|SimpleDocs,
        bokeh_scatters: dict[int, DocumentScatterResult] = None,
        bokeh_tables: dict[int, DocumentTableResult] = None,
    ):
        self.texts: list[str] = self._cast_docs(docs)
        if bokeh_scatters is None:
            self.bokeh_scatters: dict[int, DocumentScatterResult] = dict()
        else:
            self.bokeh_scatters: dict[int, DocumentScatterResult] = bokeh_scatters
        if bokeh_tables is None:
            self.bokeh_tables: dict[int, DocumentTableResult] = dict()
        else:
            self.bokeh_tables: dict[int, DocumentTableResult] = bokeh_tables

    @staticmethod
    def _cast_docs(docs: list[str]|SimpleDocs)->list[str]:
        if isinstance(docs, SimpleDocs):
            return [doc.text for doc in docs]
        else:
            return docs
    
    def get_scatter(
        self,
        n_clusters: int,
        store: bool = False,
        update: bool = False,
        labels: np.ndarray = None,
        xs: np.ndarray = None,
        ys: np.ndarray = None,
        colors: list[str] = None,
        d_labels_wamei: dict[int, str] = None,
    )->DocumentScatterResult:
        if update or n_clusters not in self.bokeh_scatters:
            scatter = DocumentScatterGenerator(
                labels = labels,
                xs = xs,
                ys = ys,
                texts = self.texts,
                nlabels = n_clusters,
                colors = colors,
                d_labels_wamei = d_labels_wamei,
            ).result
        else:
            scatter = self.bokeh_scatters[n_clusters]
        if store:
            self.bokeh_scatters[n_clusters] = scatter
        return scatter
    
    def get_scatter_from_dm(
        self,
        dm: DocumentMap,
        n_clusters: int,
        d_labels_wamei: dict[int, str] = None,
        store: bool = False,
        update: bool = False,
    )->DocumentScatterResult:
        scatter = self.get_scatter(
                n_clusters = n_clusters,
                store = store,
                update = update,
                labels = dm.dict_labels[n_clusters],
                xs = dm.xs,
                ys = dm.ys,
                d_labels_wamei = d_labels_wamei,
        )
        return scatter
    
    def get_table(
        self,
        n_clusters: int,
        labels: np.ndarray = None,
        texts: list[str] = None,
        d_labels_wamei = None,
        show_empty_table = False,
        store: bool = False,
        update: bool = False,
        empty_labels: list[int] = None,
        empty_texts: list[str] = None,
    )->DocumentTableResult:
        empty_labels = default_empty_labels if empty_labels is None else empty_labels
        empty_texts = default_empty_texts if empty_texts is None else empty_texts
        if update or n_clusters not in self.bokeh_tables:
            table = DocumentTableGenerator(
                labels = labels,
                texts = texts,
                d_labels_wamei = d_labels_wamei,
                show_empty_table = show_empty_table,
                empty_labels = empty_labels,
                empty_texts = empty_texts,
            ).result
        else:
            table = self.bokeh_tables[n_clusters]
        if store:
            self.bokeh_tables[n_clusters] = table
        return table

    def get_table_from_dm(
        self,
        dm: DocumentMap,
        n_clusters: int,
        d_labels_wamei: dict[int, str] = None,
        store: bool = False,
        update: bool = False,
        show_empty_table = False,
        empty_labels: list[int] = None,
        empty_texts: list[str] = None,
    )->DocumentTableResult:
        table = self.get_table(
            n_clusters = n_clusters,
            store = store,
            update = update,
            d_labels_wamei = d_labels_wamei,
            labels = dm.dict_labels[n_clusters],
            texts = self.texts,
            show_empty_table = show_empty_table,
            empty_labels = empty_labels,
            empty_texts = empty_texts,
        )
        return table
    
    def get_d_labels_to_texts(self, labels: np.ndarray|list[int])->dict[int, list[str]]:
        if isinstance(labels, list):
            labels = np.array(labels)
        labels_unique = np.unique(labels).tolist()
        d_labels_to_texts = {label:[] for label in labels_unique}
        for label, text in zip(labels, self.texts):
            d_labels_to_texts[label].append(text)
        return d_labels_to_texts
    
    @staticmethod
    def link_to_table(
        selector_bokeh_objects: DocumentScatterResult,
        viewer_bokeh_objects: DocumentTableResult,
        link_by: Literal["indices", "label"] = "indices",
        d_labels_to_texts: dict[int, list[str]] = None,
    ):
        viewer_bokeh_source = viewer_bokeh_objects.source
        selector_bokeh_source = selector_bokeh_objects.source
        args_for_selection_callback = dict(
            selector_source = selector_bokeh_source,
            selector_plot = selector_bokeh_objects.scatter,
            viewer_source = viewer_bokeh_source,
            initial_labels = viewer_bokeh_objects.empty_labels,
            initial_texts = viewer_bokeh_objects.empty_texts,
            d_labels_to_texts = d_labels_to_texts,
        )
        if link_by == "indices":
            code_for_selection_callback = """
            let selectedIndices = selector_source.selected.indices;
            viewer_source.data.text = selectedIndices.map(i => selector_source.data.text[i]);
            viewer_source.data.label = selectedIndices.map(i => selector_source.data.label[i]);
            viewer_source.data.label_wamei = selectedIndices.map(i => selector_source.data.label_wamei[i]);
            if (viewer_source.data.text.length == 0) {
                viewer_source.data.text = initial_texts;
                viewer_source.data.label = initial_labels;
            }
            viewer_source.change.emit();
            """
            selector_bokeh_source.selected.js_on_change(
                "indices",
                CustomJS(
                    args = args_for_selection_callback,
                    code = code_for_selection_callback
                )
            )
        elif link_by == "label":
            selector_bokeh_plot = selector_bokeh_objects.scatter
            # selector_bokeh_plot の circle に対して、tap イベントを設定する
            # tap した点の index を取得し、 d_labels_to_texts からテキストを取得し、 viewer_source に反映する
            # tap したときに、他のグループの circle を mute する

            code_for_selection_callback = """
            let tappedIndex = selector_source.selected.indices;
            let selectedLabel = selector_source.data.label[tappedIndex];
            let selectedLabelWamei = selector_source.data.label_wamei[tappedIndex];
            let selectedTexts = d_labels_to_texts[selectedLabel];
            viewer_source.data.text = selectedTexts;
            // viewer_source.data.label には selectedTexts の長さ分だけ selectedLabel を入れる
            let selectedLabels = Array(selectedTexts.length).fill(selectedLabel);
            let selectedLabelWameis = Array(selectedTexts.length).fill(selectedLabelWamei);
            viewer_source.data.label = selectedLabels;
            viewer_source.data.label_wamei = selectedLabelWameis;
            if (viewer_source.data.text.length == 0) {
                viewer_source.data.text = initial_texts;
                viewer_source.data.label = initial_labels;
            }
            viewer_source.change.emit();
            selector_plot.change.emit();
            """
            callback_for_tap = CustomJS(
                args = args_for_selection_callback,
                code = code_for_selection_callback
            )
            selector_bokeh_plot.js_on_event(
                "tap",
                callback_for_tap
            )
        else:
            raise ValueError(f"link_by must be either 'indices' or 'label', but {link_by} was given")
        return None
