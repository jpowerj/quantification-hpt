import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PointData:
    def __init__(self, x, y, label, anchor, xshift=0.0, yshift=0.0,
                 formatted=None, draw_point=True, draw_label=True):
        self.x = x
        self.y = y
        self.label = label
        self.anchor = anchor
        self.xshift = xshift
        self.yshift = yshift
        self.formatted = formatted
        self.draw_point = draw_point
        self.draw_label = draw_label

    def escaped_label(self):
        #return self.label.replace("_", "\\_")
        return self.label.replace("_", "-")

    def __str__(self):
        return f"PointData[x={self.x},y={self.y},label={self.label},anchor={self.anchor}]"

    def pgf_str(self):
        #quoted_label = "\"" + escaped_label + "\""
        #styled_label = "\\entpgf{" + escaped_label + "}"
        if self.draw_point:
            return f"({self.x}, {self.y})[]"
        return ""

    def point_str(self, debug=False):
        if debug:
            print("[point_str()] self.label = " + str(self.label))
        point_options = ""
        if not self.draw_point:
            # "Invisible" point
            point_options = "[inner sep=0] "
        return "\\node " + point_options + "(" + self.escaped_label() + ") at (axis cs:" + str(self.x) + ", " + str(self.y) + "){};"

    def label_str(self):
        label_pos = "\\node[anchor = " + self.anchor + ", xshift=" + str(self.xshift) + ", yshift=" + str(self.yshift) + "] (" + self.escaped_label() + "l) at(axis cs: " + str(self.x) + ", " + str(self.y) + ")"
        default_formatted = "$\\entpgf{" + self.escaped_label() + "}$"
        if self.formatted is None:
            label_formatted = default_formatted
        else:
            label_formatted = self.formatted
        label_text = "{" + label_formatted + "};"
        if self.draw_label:
            return label_pos + label_text
        return ""

class Arrow:
    def __init__(self, start, end, dash_str, type_str="->"):
        self.start = start.replace("_", "\\_")
        self.end = end.replace("_", "\\_")
        self.dash_str = dash_str
        if self.dash_str != "" and (not self.dash_str.endswith(",")):
            self.dash_str = self.dash_str + ","
        print("[Arrow()] self.dash_str = " + str(self.dash_str))
        self.type_str = type_str

    def __str__(self):
        latex_option = ",-latex"
        if self.type_str != "->":
            latex_option = ""
        return "\\draw[" + self.type_str + "," + self.dash_str + ">=stealth,thick," + latex_option + "](" + self.start + ") to (" + self.end + ");"

tikz_start = """
\\begin{tikzpicture}
\\pgfplotsset{ticks=none}
		\\begin{axis}[
		    axis x line=bottom,
			axis y line=left,
			xmin=!pyxmin!-!pyxcap!, xmax=!pyxmax!+!pyxcap!,
			ymin=!pyymin!-!pyycap!, ymax=!pyymax!+!pyycap!,
			xtick={!pyxmin!,!pyxmax!},ytick={!pyymin!,!pyymax!},
			xlabel=!pyxlabel!,ylabel=!pyylabel!,
			%x label style={anchor=west},
			%y label style={anchor=south},
			width=\\textwidth
			]
			\\addplot+[mark options={fill=black,color=black},only marks,point meta=explicit symbolic, nodes near coords] coordinates {
"""
tikz_end_addplot = """
};
"""
tikz_end = """
\\end{axis}
\\end{tikzpicture}
"""

tikz_header = """
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{mathtools}
\\usepackage{fullpage}
\\usepackage[T1]{fontenc}
\\usepackage{lmodern}
\\usepackage{tikz}
\\usetikzlibrary{calc,intersections,through,backgrounds}
\\usetikzlibrary{bayesnet}
\\usepackage{tikzscale}
\\usepackage{tkz-euclide}
\\usepackage{tcolorbox}
\\tcbuselibrary{skins,breakable}
% pgfplots
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.8}
% For entities in pgfplots
\\newcommand{\entpgf}[1]{\\texttt{#1}}
"""

def floor_half(num):
    return np.floor(num * 2) / 2

def ceil_half(num):
    return np.ceil(num * 2) / 2

def get_anchor(token, label_data):
    if token in label_data and 'anchor' in label_data[token]:
        return label_data[token]['anchor']
    return "north"

def get_xshift(token, label_data):
    if token in label_data and 'xshift' in label_data[token]:
        return label_data[token]['xshift']
    return 0.0

def get_yshift(token, label_data):
    if token in label_data and 'yshift' in label_data[token]:
        return label_data[token]['yshift']
    return 0.0

def get_prop(token, label_data, prop_name, default_val=None):
    if token in label_data and prop_name in label_data[token]:
        return label_data[token][prop_name]
    return default_val

def custom_latex_export(df, label_data=None, arrow_data=None, floor_ceil=False,
                        self_contained=False, pad_pct=0.05, axis_options=None,
                        debug=False):
    vprint = print if debug else lambda x: None
    if label_data is None:
        label_data = {}
    if axis_options is None:
        xlabel = '$x$'
        ylabel = '$y$'
    else:
        xlabel = axis_options['xlabel']
        ylabel = axis_options['ylabel']
    pgf_list = []
    point_list = []
    label_list = []
    for row_index, row in df.iterrows():
        cur_token = row_index
        vprint(f"Adding {cur_token}")
        should_include = get_prop(cur_token, label_data, 'include', True)
        if not should_include:
            continue
        cur_anchor = get_anchor(cur_token, label_data)
        cur_formatted = get_prop(cur_token, label_data, 'formatted')
        cur_xshift = get_xshift(cur_token, label_data)
        cur_yshift = get_yshift(cur_token, label_data)
        cur_draw_point = get_prop(cur_token, label_data, 'draw_point', True)
        cur_draw_label = get_prop(cur_token, label_data, 'draw_label', True)
        cur_pd = PointData(x=row['x'], y=row['y'], label=cur_token,
                           formatted=cur_formatted,
                           anchor=cur_anchor,
                           xshift=cur_xshift, yshift=cur_yshift,
                           draw_point=cur_draw_point, draw_label=cur_draw_label)
        if debug:
            print("cur_pd (PointData): " + str(cur_pd))
        pgf_list.append(cur_pd.pgf_str())
        # Point str
        cur_point_str = cur_pd.point_str(debug=debug)
        point_list.append(cur_point_str)
        label_list.append(cur_pd.label_str())
    pgf_str = "\n".join(pgf_list)
    point_str = "\n".join(point_list)
    label_str = "\n".join(label_list)
    pyxmin = df['x'].min()
    if floor_ceil:
        pyxmin = floor_half(pyxmin)
    pyxmax = df['x'].max()
    if floor_ceil:
        pyxmax = ceil_half(pyxmax)
    pyymin = df['y'].min()
    if floor_ceil:
        pyymin = floor_half(pyymin)
    pyymax = df['y'].max()
    if floor_ceil:
        pyymax = ceil_half(pyymax)
    tex_start_lims = tikz_start
    tex_start_lims = tex_start_lims.replace("!pyxmin!", str(pyxmin)).replace("!pyxmax!", str(pyxmax))
    tex_start_lims = tex_start_lims.replace("!pyymin!", str(pyymin)).replace("!pyymax!", str(pyymax))
    # Axis labels
    tex_start_lims = tex_start_lims.replace("!pyxlabel!", xlabel)
    tex_start_lims = tex_start_lims.replace('!pyylabel!', ylabel)
    # Extend the axis a bit past the min/max
    pyyrange = pyymax - pyymin
    pyycap = pad_pct * pyyrange
    tex_start_lims = tex_start_lims.replace("!pyycap!", str(pyycap))
    pyxrange = pyxmax - pyxmin
    pyxcap = pad_pct * pyxrange
    tex_start_lims = tex_start_lims.replace("!pyxcap!", str(pyxcap))
    return_str = tex_start_lims + pgf_str + tikz_end_addplot
    # Now the labels
    return_str = return_str + point_str + "\n" + label_str + "\n"
    # And arrows, if any
    if arrow_data is not None:
        for cur_arrow in arrow_data:
            arrow_obj = Arrow(cur_arrow['pointA'], cur_arrow['pointB'], cur_arrow['linestyle'], cur_arrow['arrowtype'])
            arrow_str = str(arrow_obj)
            return_str = return_str + arrow_str + "\n"
    return_str = return_str + tikz_end
    if self_contained:
        # Wrap it in a document
        doc_start = "\\documentclass{article}\n" + tikz_header + "\n\\begin{document}\n"
        doc_end = "\n\\end{document}"
        wrap_str = doc_start + return_str + doc_end
        return wrap_str
    return return_str
# For previewing
#tex_str = custom_latex_export(lev_df, self_contained=True)
#pyperclip.copy(tex_str)
# For exporting
#tex_str = custom_latex_export(lev_df)
#pyperclip.copy(tex_str)
#tex_output_fpath = os.path.join(fig_path, "hobbes_embeddings.tex")
#with open(tex_output_fpath, 'w', encoding='utf-8') as outfile:
#    outfile.write(tex_str)

# Seaborn

def draw_arrow(df, tok1, tok2, astyle=None, lstyle=None):
    if "token" in df.columns:
        df_temp = df.set_index("token").copy()
    else:
        df_temp = df
    x1 = df_temp.loc[tok1,"x"]
    x2 = df_temp.loc[tok2,"x"]
    y1 = df_temp.loc[tok1,"y"]
    y2 = df_temp.loc[tok2,"y"]
    #plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=1)
    dx = x2 - x1
    dy = y2 - y1
    #plt.arrow(x1,y1,dx,dy,width=0.01,head_width=0.1,head_length=0.1,length_includes_head=True,color='black')
    # This goes from xytext to xy...
    # Possible arrowstyle values: "->", "-[", "|-|"
    if astyle is None:
        astyle = "->"
    if lstyle is None:
        lstyle = "-"
    plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle=astyle, linestyle=lstyle, color='black'))

def label_points(x, y, val, ax, x_offset=0.015, adjustments=None):
    # Val might be an index, so convert it
    if type(val) != pd.Series:
        val = val.to_series()
    label_df = pd.concat({'x': x, 'y': y, 'label': val}, axis=1)
    if adjustments:
        # And adjust the vals accordingly
        label_df.set_index('label', inplace=True)
        for adj_word, adj_shift in adjustments.items():
            if adj_word in label_df.index:
                label_df.loc[adj_word] = label_df.loc[adj_word] + adj_shift
        label_df.reset_index(inplace=True)
    for i, point in label_df.iterrows():
        ax.text(point['x']+x_offset, point['y'], str(point['label']), va='center', fontdict={'size':13})