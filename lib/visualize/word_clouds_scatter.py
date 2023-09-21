from typing import Iterator
import re
import numpy as np
import pandas as pd

from bokeh.plotting import figure, save
from bokeh.models import BoxZoomTool, TapTool, WheelZoomTool, PanTool, ResetTool, ColumnDataSource, Text, CustomJS, WheelPanTool, Range1d, HoverTool
from bokeh.models import Div
from bokeh.plotting.figure import Figure
from bokeh.palettes import Category10, Category20
from wordcloud import WordCloud

class WordCloudsScatterResult:
    def __init__(
        self,
        source: ColumnDataSource,
        scatter: Figure,
    ):
        self.source: ColumnDataSource = source
        self.scatter: Figure = scatter

class WordCloudsScatterPositions:
    def __init__(
            self,
            xs_points: np.ndarray,
            ys_points: np.ndarray,
            labels_points: np.ndarray,
            labels_unique: list[str],
            d_labels_wamei: dict[str, str] = None,
            ):
        magic_scale_factor = 0.4
        xs_cloud = []
        ys_cloud = []
        sizes_cloud = []
        labels_points = np.array(labels_points)
        for label in labels_unique:
            indices = np.where(labels_points == label)[0]
            xs_cloud.append(xs_points[indices].mean())
            ys_cloud.append(ys_points[indices].mean())
            sizes_cloud.append(len(indices) / len(labels_points))
        self.xs_cloud = np.array(xs_cloud)
        self.ys_cloud = np.array(ys_cloud)
        self.sizes_cloud = np.array(sizes_cloud)
        # xs_cloud, ys_cloud, を 0 から 1 の範囲にスケーリングする
        self.xs_cloud = (self.xs_cloud - self.xs_cloud.min()) / (self.xs_cloud.max() - self.xs_cloud.min())
        self.ys_cloud = (self.ys_cloud - self.ys_cloud.min()) / (self.ys_cloud.max() - self.ys_cloud.min())
        self.sizes_cloud = self.sizes_cloud / self.sizes_cloud.max()
        self.labels_unique = labels_unique
        self.d_labels_wamei = d_labels_wamei
    
    @property
    def labels_wamei(self):
        if self.d_labels_wamei is None:
            return self.labels_unique
        else:
            return [f"{label}:{self.d_labels_wamei.get(label, '')}" for label in self.labels_unique]

class WordCloudsScatter:
    def __init__(
            self,
            labels_unique: list[str],
            d_vectors: dict[int|str, tuple[str, np.float64]],
            xs: np.ndarray,
            ys: np.ndarray,
            labels: np.ndarray,
            array_size: int = 300,
            fill_alpha = 0.07,
            line_alpha = 0.6,
    ):
        self.labels_unique = labels_unique
        self.d_vectors = d_vectors
        self.positions = WordCloudsScatterPositions(xs, ys, labels, labels_unique)
        self.array_size: int = array_size
        self.wordclouds = self._generate_wordclouds()
        self.fill_alpha = fill_alpha
        self.line_alpha = line_alpha

    @property
    def d_labels_wamei(self):
        return self.positions.d_labels_wamei
    
    @d_labels_wamei.setter
    def d_labels_wamei(self, d_labels_wamei:dict[str, str]):
        self.positions.d_labels_wamei = d_labels_wamei

    @staticmethod
    def circle(size = 300):
        d = size
        r = size // 2
        m = r * 0.86
        x, y = np.ogrid[:d, :d]
        mask = (x - r) ** 2 + (y - r) ** 2 > m ** 2
        return 255 * mask.astype(int)
    
    def get_cmap(self, nlabels: int)->dict[str, str]:
        return Category10[nlabels] if nlabels <= 10 else Category20[nlabels]

    def _generate_wordclouds(self)->list[WordCloud]:
        wordclouds = []
        for label in self.labels_unique:
            wc = WordCloud(
                background_color="white",
                mask=WordCloudsScatter.circle(self.array_size),
                font_path="C:\Windows\Fonts\meiryo.ttc",                                              
            )
            word_freqs = {vocab : tfidf_value for vocab, tfidf_value in self.d_vectors[label]}
            wordclouds.append(wc.generate_from_frequencies(word_freqs))
        return wordclouds
    
    def to_dataframe(self)->pd.DataFrame:
        def parse_line(line, label = None)->list[int|str]:
            line_pattern = re.compile(r'<text transform="translate\((\d+),(\d+)\)( rotate\((-\d+)\))?" font-size="(\d+)" style="fill:rgb\((\d+), (\d+), (\d+)\)">(.+)</text>')
            m = line_pattern.match(line)
            if m:
                x = int(m.group(1))
                y = int(m.group(2))
                angle = int(m.group(4)) if m.group(4) else 0
                font_size = int(m.group(5))
                red = int(m.group(6))
                green = int(m.group(7))
                blue = int(m.group(8))
                text = m.group(9)
                if label is not None:
                    return x, y, font_size, red, green, blue, text, angle, label
                else:
                    return x, y, font_size, red, green, blue, text, angle
            else:
                return None
        def svg_to_list(svg_text:str, label = None)->list[list[int|str]]:
            lines = svg_text.split('\n')
            lines = [parse_line(line, label) for line in lines]
            lines = [line for line in lines if line]
            return lines
        def convert_rgb_to_hex(r:pd.Series, g:pd.Series, b:pd.Series):
            return "#" + pd.Series([f"{r.iloc[i]:02x}{g.iloc[i]:02x}{b.iloc[i]:02x}" for i in range(len(r))])        
        
        svg_data = []
        for label, wc in zip(self.labels_unique, self.wordclouds):
            svg_text = wc.to_svg()
            lines = svg_to_list(svg_text, label = label)
            svg_data += lines
        df = pd.DataFrame(svg_data, columns=["x", "y", "font-size", "red", "green", "blue", "text", "angle", "label"])
        df["color"] = convert_rgb_to_hex(df["red"], df["green"], df["blue"])
        return df
    
    def to_bokeh(
            self,
            canvas_size:tuple[int, int] = (1000, 600),
            size_factor = 0.06
        )->WordCloudsScatterResult:
        canvas_ratio = canvas_size[0] / canvas_size[1]
        magic_scale_factor = 1.04
        magic_number = 50
        df_word_cloud = self.to_dataframe()
        df_word_cloud["angle"] = df_word_cloud["angle"] / 180 * np.pi * -1
        # 900 x 900 のうち、[50, 850] x [50, 850] に収まっている
        original_margin = 50
        original_xoffset = 450
        original_yoffset = 450
        original_canvas_diameter = 900
        original_diameter = 900 - original_margin*2
        original_maximum_font_size = df_word_cloud["font-size"].max()
        df_word_cloud["y"] = original_canvas_diameter - df_word_cloud["y"]
        diameter_list = []
        
        def cloud_position_to_canvas_position(x, y, canvas_size, magic_number):
            x_screen = (x - 0.5) * (canvas_size[0] - 2 * magic_number)*0.65 + (canvas_size[0] - 2 * magic_number)/2
            y_screen = (y - 0.5) * (canvas_size[1] - 2 * magic_number)*0.65 + (canvas_size[1] - 2 * magic_number)/2
            return x_screen, y_screen
        
        for ilabel, (label, wc) in enumerate(zip(self.labels_unique, self.wordclouds)):
            x_offset, y_offset = cloud_position_to_canvas_position(self.positions.xs_cloud[ilabel], self.positions.ys_cloud[ilabel], canvas_size, magic_number)
            size = self.positions.sizes_cloud[ilabel] * canvas_size[0] * size_factor / original_maximum_font_size # 最大でも 8% 程度のフォントサイズ
            radius = canvas_size[1] * size / 2
            diameter_list.append(radius * 2)
            scale_space = radius/(original_diameter/2)
            scale_font = scale_space * magic_scale_factor

            indices = df_word_cloud["label"] == label
            df_word_cloud.loc[indices, "x"] = (df_word_cloud.loc[indices, "x"] - original_xoffset)* scale_space + x_offset
            df_word_cloud.loc[indices, "y"] = (df_word_cloud.loc[indices, "y"] - original_yoffset)* scale_space + y_offset
            df_word_cloud.loc[indices, "font-size"] = df_word_cloud.loc[indices, "font-size"].apply(lambda x:f"{x * scale_font:.1f}px")
        df_word_cloud = df_word_cloud[["x", "y", "text", "angle", "color", "font-size"]]
        df_word_cloud["text_font"] = "Meiryo"
        df_word_cloud["text_align"] = "left"
        df_word_cloud["text_baseline"] = "center"

        source = ColumnDataSource(df_word_cloud)
        xsize_initial = canvas_size[0]
        x_range_initial = (0, canvas_size[0] - magic_number)
        y_range_initial = (0, canvas_size[1] - magic_number)
        max_radius = max(diameter_list) / 2
        extend = (max_radius + magic_number) * max(canvas_ratio, 1/canvas_ratio)
        x_range = Range1d(
            start = x_range_initial[0],
            end = x_range_initial[1],
            bounds = (-extend*5, canvas_size[0] + extend*5)
        )
        y_range = Range1d(
            start = y_range_initial[0],
            end = y_range_initial[1],
            bounds = (-extend*5, canvas_size[1] + extend*5)
        )
        # p = figure(tooltips=None, tools=tools, match_aspect=True, title="word clouds")

        p = figure(tooltips=None,
                tools="wheel_zoom,pan,tap,reset",
                active_drag = "pan",
                active_scroll = "wheel_zoom",
                active_tap = "tap",
                match_aspect=True,
                x_range=x_range, y_range=y_range,
                width=canvas_size[0], height=canvas_size[1], 
                title="グループごとの特徴的な単語のワードクラウド(クリックすると本文が下に表示されます)",
                )

        glyph = Text(
            x="x", y="y", 
            # x = "zero", y = "zero", x_offset = "x", y_offset = "y",
            text="text", angle="angle", text_color="color", text_font_size = "font-size", 
            text_align="text_align", text_baseline="text_baseline",
            text_font="text_font",
        )

        # cloud の中心に円を書く
        # x_offset, y_offset = cloud_position_to_canvas_position(self.positions.xs_cloud[ilabel], self.positions.ys_cloud[ilabel], canvas_size, magic_number)
        xs_cloud, ys_cloud = cloud_position_to_canvas_position(self.positions.xs_cloud, self.positions.ys_cloud, canvas_size, magic_number)
        labels = self.positions.labels_unique
        nlabels = len(labels)
        cmap = self.get_cmap(nlabels)
        labels_wamei = self.positions.labels_wamei
        source_cloud = ColumnDataSource(dict(
            x = xs_cloud,
            y = ys_cloud,
            diameter = [diameter *1.05 for diameter in diameter_list],
            label_wamei = labels_wamei,
            label = labels,
            color = [cmap[label] for label in labels]
        ))

        cloud_invisible_markers = p.circle(
            x="x", y="y",
            source=source_cloud, size="diameter", color="color", 
            legend_group ="label_wamei",
            fill_alpha=self.fill_alpha, line_alpha=self.line_alpha, line_width=2.0,
            )
        
        from bokeh.models.glyphs import Circle
        cloud_invisible_markers.selection_glyph = Circle(
            fill_alpha=(self.fill_alpha) ** 0.8,
            line_alpha=(self.line_alpha) ** 0.8, 
            fill_color = "color",
            line_color = "color",
            line_width=3.0,
        )
        cloud_invisible_markers.nonselection_glyph = Circle(
            fill_alpha=(self.fill_alpha) ** 1.2,
            line_alpha=(self.line_alpha) ** 1.2,
            fill_color = "color",
            line_color = "color",
            line_width=1.5,
        )

        p.add_glyph(source, glyph)

        hover = HoverTool(
            tooltips=[
                ("グループ名", "@label_wamei"),
            ],
            mode="mouse",
            renderers=[cloud_invisible_markers],
        )
        p.add_tools(hover)

        p.toolbar.logo = None
        p.toolbar_location = None
        # ticks を消す
        p.xaxis.visible = False
        p.yaxis.visible = False

        # grid を消す
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

        # zoom したときに CustomJS で glyph の font-size を変更して、zoom に合わせて文字の大きさを変える
        args_for_js = dict(
            source=source,
            source_cloud=source_cloud,
            diameter_initial = diameter_list,
            plot=p,
            xsize_initial = xsize_initial - magic_number,
            fontsize_initial = df_word_cloud["font-size"].to_list()
            )
        js = """
        let scale = (xsize_initial)/(plot.x_range.end - plot.x_range.start);
        // fontSize は 13.2px のような文字列であり、px を取り除いた数値部分を scale 倍して px をつけて文字列に戻す。数値は小数点以下第5位までに丸める
        let newFontSize = fontsize_initial.map(x => `${(parseFloat(x) * scale).toFixed(5)}px`);

        source.data["font-size"] = newFontSize;
        source_cloud.data["diameter"] = diameter_initial.map(x => x * scale);
        source.change.emit();
        source_cloud.change.emit();
        """

        p.js_on_event("wheel", CustomJS(args=args_for_js, code=js))
        p.js_on_event("boxzoom", CustomJS(args=args_for_js, code=js))
        
        return WordCloudsScatterResult(source_cloud, p)


    def __iter__(self)->Iterator[WordCloud]:
        return iter(self.wordclouds)
    
    def __getitem__(self, index: int|str)->WordCloud:
        return self.wordclouds[index]

    def __len__(self)->int:
        return len(self.wordclouds)
    
    def to_dict(self)->dict[str, WordCloud]:
        return {label: wc for label, wc in zip(self.labels_unique, self.wordclouds)}
    
