# Python imports
from collections import Counter
import glob
from multiprocessing import Pool
import os
import sys

# 3rd party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from gensim.models import Word2Vec
import joblib
from tqdm import tqdm
from nltk import WordNetLemmatizer
from sklearn.manifold import TSNE

# Local imports
import hptutil

tqdm.pandas()


def set_seaborn_style():
    # Idk why we have to call this twice for it to work but... here we are
    sns.set(rc={'figure.figsize': (8, 5)})
    sns.set_theme(style="whitegrid")


set_seaborn_style()

global_window_high = 21
global_window_low = 9
global_epochs = 80
global_min_count = 10
global_workers = 8
fig_path = os.path.join("..", "figs")

token_sets = [
    ["black", "colored", "n*****", "nigger", "niggers", "negro", "negros", "negroes", "blacks"],
    ["child", "children", "childs", "chillun"],
    ["coop", "coops"],
    ["field", "fields"],
    # ["free"]
    ["house", "houses"],
    ["man", "men", "mans"],
    ["master", "marster", "masters", "marsters", "marse", 'massa'],
    # ["quarters","quarter","coop","coops"],
    ["slave", "slaves"],
    ["white", "whites"],
    ["woman", "women"],
    ["exslave", "exslaves"],
    ["slavery", "slavry"],
    ['darky', 'darkey'],
    ['thing', 'ting'],
    ['have', 'hab'],
    ['live', 'lib'],
    ['was', 'wus'],
    ['folk', 'folks'],
]
token_map = {}
for cur_token_set in token_sets:
    first_token = cur_token_set[0]
    for cur_other_token in cur_token_set[1:]:
        token_map[cur_other_token] = first_token

stopwords_en = nltk.corpus.stopwords.words('english')
stopwords_custom = [
    # Alphabetical
    'abc', 'ah', 'ai', 'around', 'atter', 'aw', 'can', 'come', 'dar', 'dat', 'day', 'de', "dey", 'em', 'er',
    'every', 'fa', 'far',
    'go', 'got', 'jes', 'know', 'move',
    'n', 'none', 'nt',
    'oh', 'old', 'ole', 'p', 'say', 'see', 'still', 'tell', 'th', 'time', 'try',
    'us', 'uv', 'uz', 'w', "war", 'would', 'wuz', 'year', 'yo', 'yuh',
    '$', '_',
    # Unordered
    'make', 'never', 'take', 'could', 'bout', 'live', 'en', 'wid', 'big', 'back',
    'dem', 'give', 'place', 'ter', 'bear', 'get', 'like', 'little', 'do', 'use', 'long',
    'well', 'den', 'two', 'dere', 'way', 'thing', 'much', 'put', 'ge', 'uh',
    'call', 'git', 'run', 'keep', 'cause', 'think', 'remember', 'wus',
    'look', 'away', 'let', 'yes', 'many', 'dis', 'till', 'ever', 'lot', 'first', 'sho',
    'nothin', 'always', 'member', 'round', 'sometimes', 'not', 'fer', 'lak',
    # months
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
    'october', 'november', 'december',
    # numbers
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
    'eighteen', 'nineteen', 'twenty',
    'thirtysix', 'fifty', 'fiftyfive', 'eighty', 'eightytwo', 'eightyseven', 'ninety',
    'once', 'twice', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
]
stopwords_en_full = stopwords_en + stopwords_custom


def load_text(fpath):
    with open(fpath, 'r', encoding='utf-8') as infile:
        text = infile.read()
    return text


def lemmed(text, cores=4):  # tweak cores as needed
    with Pool(processes=cores) as pool:
        wnl = WordNetLemmatizer()
        result = pool.map(wnl.lemmatize, text)
    return result

punct_chars = ".,\"':!?;()“”’`&[]1234567890–§-—"
def remove_punct(token):
    return "".join([t for t in token if t not in punct_chars])


def clean_text(text, lemmatizer):
    all_sent_tokens = []
    text_sents = nltk.sent_tokenize(text, "english")
    for cur_sent in text_sents:
        sent_tokens = nltk.word_tokenize(cur_sent, "english")
        clean_tokens = [t.lower() for t in sent_tokens]
        clean_tokens = [remove_punct(t) for t in clean_tokens]
        clean_tokens = [t for t in clean_tokens if t not in stopwords_en_full]
        clean_tokens = [t for t in clean_tokens if len(t) > 0]
        clean_tokens = [(token_map[t] if t in token_map else t) for t in clean_tokens]
        # Lastly, lemmatize using spacy
        # clean_sent = " ".join(clean_tokens)
        # clean_doc = spacy_en(clean_sent)
        # clean_lemmas = [cur_token.lemma_ for cur_token in clean_doc]
        clean_lemmas = [lemmatizer.lemmatize(t) for t in clean_tokens]
        all_sent_tokens.append(clean_lemmas)
    return all_sent_tokens


# Cluster genders
gend_male = [
    'man',
    'boy',
    'father',
    'son',
    'husband',
    'grandson',
    'pa', 'papa', 'pappy',
    'brothers',
    'brother',
    'grandpa',
    'grandfather',
    'soninlaw',
    'stepfather',
    'gentleman',
    'mr', 'mister',
    'uncle',
]
gen_male_ext = [
    'roger', 'alfred', 'randall', 'billy', 'jake', 'nicholas', 'jim',
    'col', 'colonel', 'daddy',
]
gend_female = [
    'woman',
    'miss',
    'girl',
    'wife',
    'mother',
    'mammy',
    'daughter',
    'sisters',
    'sister',
    'grandma',
    'grandmother',
    'missus',
    'granddaughter',
    'mistress', 'mistess',
    'maam', 'ms', 'mrs',
    'aunt',
    'mama',
    'gal',
]
gen_female_ext = [
    'ann', 'lucy', 'nellie', 'maggie', 'elizabeth', 'mary', 'mamie', 'marse',
]
map_gend = {token: 'man' for token in gend_male}
map_gend.update({token: 'woman' for token in gend_female})


def get_mapped_token(word, replacement_map):
    return replacement_map[word] if word in replacement_map else word


def replace_word_tokens(word_tokens, replacement_map):
    return [get_mapped_token(w, replacement_map) for w in word_tokens]


def remove_word_tokens(word_tokens, remove_list):
    return [w for w in word_tokens if w not in remove_list]


def replace_sent_tokens(orig_sent_tokens, replacement_map):
    new_sent_tokens = [replace_word_tokens(sent, replacement_map) for sent in orig_sent_tokens]
    return new_sent_tokens


def remove_sent_tokens(orig_sent_tokens, remove_list):
    new_sent_tokens = [remove_word_tokens(sent, remove_list) for sent in orig_sent_tokens]
    # Remove empty sents
    new_sent_tokens = [t for t in new_sent_tokens if len(t) > 0]
    return new_sent_tokens


# (From the diss)
kw_main_orig = ["white", "slave", "master", "black", "man", "woman", ]
# "coop","house",
# "field",
# "town",
# "yard",
# "land",
#            "prison"]
# (New)
kw_tsne = [
    "white", "black", "slave", "master", "man", "woman",
    # "rich","poor",
    "slavery", "freedom"
]
kw_main = kw_main_orig + [
    # "plantation",
    # "jail",
    # "free",
    # "mother", "father", "child",
    # "field",
    # "church",
    # "whip"
    # "rich",
    # "poor",
]
kw_full = kw_main + [
    "cotton",
    # "colored",
    # "n*****",
    "slavery",
    "freedom",
    "whip",
    "sold",
    "good",
    "plantation",
    "overseer",
    "beat",
    "north", "south",
    "care",
    "house", "field",
    "child",
    "cook", "clean",
    "money"
]


def get_embedding_df(w2v_model, kw_list, tagged=False):
    w2v_rows = []
    emb_keywords = kw_list
    if tagged:
        emb_keywords = [kw + "[m]" for kw in kw_list] + [kw + "[h]" for kw in kw_list]
    for cur_kw in emb_keywords:
        if cur_kw not in w2v_model.wv:
            continue
        cur_vec = w2v_model.wv[cur_kw]
        cur_row = [cur_kw] + list(cur_vec)
        w2v_rows.append(cur_row)
    w2v_df = pd.DataFrame(w2v_rows, columns=['token', 'x', 'y'])
    return w2v_df


def combine_sent_tokens(sent_tokens):
    tokens_combined = []
    for cur_sent_tokens in sent_tokens:
        tokens_combined.extend(cur_sent_tokens)
    return tokens_combined

def contract_midpoint(df, token, proportion):
    for dimension in ['x', 'y']:
        orig_point = df.loc[token, dimension]
        centroid = df.loc['centroid', dimension]
        midpoint = proportion * centroid + (1 - proportion) * orig_point
        df.at[token, dimension] = midpoint

def get_token_counts(tokens):
    clean_tokens_count = Counter(tokens)
    counts_sorted = sorted(clean_tokens_count.items(), key=lambda x: x[1], reverse=True)
    return counts_sorted

skip_cleaning = True
verbose = True

def clean_wpa_texts():
    # Loading
    state_paths = glob.glob("../WPA_Online_Appendix/Narratives by State/*")
    wpa_texts = []
    for cur_path in state_paths:
        cur_state = os.path.basename(cur_path)
        for cur_subdir in ["Black", "Unidentified", "White"]:
            subdir_path = os.path.join(cur_path, cur_subdir)
            subdir_fpaths = glob.glob(os.path.join(subdir_path, "*"))
            for cur_subdir_fpath in subdir_fpaths:
                file_data = dict()
                file_data['text_raw'] = load_text(cur_subdir_fpath)
                file_data['fname'] = os.path.basename(cur_subdir_fpath)
                file_data['race'] = cur_subdir
                file_data['state'] = cur_state
                wpa_texts.append(file_data)
    wpa_df = pd.DataFrame(wpa_texts)
    # Cleaning
    wnl = WordNetLemmatizer()
    clean_w_lemmatizer = lambda x: clean_text(x, wnl)
    wpa_df['text'] = wpa_df['text_raw'].progress_apply(clean_w_lemmatizer)
    wpa_df.to_pickle("wpa_text_df.pkl")
    return wpa_df

def gen_plot_svg(df, xvar, yvar, title, svg_fname, xlim=None, ylim=None,
                 label_adjustments=None,
                 arrow_data=None):
    print("Generating raw embedding plot")
    if arrow_data is None:
        arrow_data = []
    set_seaborn_style()
    plt.figure(figsize=(12, 8))
    word_plot = sns.scatterplot(data=df, x=xvar, y=yvar)
    word_plot.grid(False)
    if xlim is not None:
        word_plot.set_xlim(xlim)
    if ylim is not None:
        word_plot.set_ylim(ylim)
    sns.despine(bottom=False, left=False, right=True, top=True)
    # word_plot.set_xticks([])
    # word_plot.set_yticks([])
    word_plot.set_title(title, fontdict={'size': 14})
    hptutil.label_points(df[xvar], df[yvar], df.index.to_series(), plt.gca(),
                         x_offset=0.01, adjustments=label_adjustments)
    # Draw arrows
    for a in arrow_data:
        if 'lstyle' not in a:
            a['lstyle'] = '-'
        if 'astyle' not in a:
            a['astyle'] = '->'
        hptutil.draw_arrow(df, a['pointA'], a['pointB'], lstyle=a['lstyle'],
                           astyle=a['astyle'])
    # plt.arrow(0.99,1.76,1,1,width=0.01,head_width=0.1,head_length=0.1,length_includes_head=True,color='black')
    # plt.annotate("", xy=(0.99, 1.76), xytext=(1.96, -0.45), arrowprops=dict(arrowstyle="->",color='black'))
    plt.tight_layout()
    svg_fpath = os.path.join(fig_path, svg_fname)
    plt.savefig(svg_fpath)

def main():
    global_seed = 1948
    if len(sys.argv) > 1:
        global_seed = int(sys.argv[1])
    vprint = print if verbose else lambda x: None
    if skip_cleaning:
        wpa_df = pd.read_pickle("./wpa_text_df.pkl")
    else:
        wpa_df = clean_wpa_texts()

    # Gender clusters
    vprint("Computing gender clusters")
    corpus_sent_tokens = []
    corpus_sent_tokens_gend = []
    for row_index, row in wpa_df.iterrows():
        cur_sent_tokens = row['text']
        # print(cur_sent_tokens[:5])
        # Gender map
        cur_sent_tokens_gend = replace_sent_tokens(cur_sent_tokens, map_gend)
        # print(cur_sent_tokens[:5])
        # break
        corpus_sent_tokens.extend(cur_sent_tokens)
        corpus_sent_tokens_gend.extend(cur_sent_tokens_gend)

    # For the sake of getting individual token counts, combine into a giant list here
    all_tokens = combine_sent_tokens(corpus_sent_tokens)
    tcounts = get_token_counts(all_tokens)

    # w2v: small window
    # wpa_model_lowwin = Word2Vec(
    #     sentences=corpus_sent_tokens, vector_size=300,
    #     window=global_window_low,
    #     min_count=global_min_count, workers=8,
    #     epochs=global_epochs, seed=global_seed
    # )
    vprint("Training low-window w2v model with gender clusters...")
    wpa_model_lowwin_gend = Word2Vec(
        sentences=corpus_sent_tokens_gend, vector_size=300,
        window=global_window_low,
        min_count=global_min_count, workers=8,
        epochs=global_epochs, seed=global_seed
    )
    vprint("Low-window w2v models trained")

    # w2v: low-dim
    vprint("Training low-dim w2v model...")
    wpa_model_lowdim = Word2Vec(
        sentences=corpus_sent_tokens, vector_size=2,
        # window=global_window,
        window=global_window_low,
        min_count=global_min_count, workers=global_workers,
        # epochs=global_epochs,
        epochs=global_epochs,
        seed=global_seed,
    )
    wpa_model_lowdim.save("w2v_wpa_lowdim.model")
    vprint("Training low-dim w2v model with gender clusters...")
    wpa_model_lowdim_gend = Word2Vec(
        sentences=corpus_sent_tokens_gend, vector_size=2,
        # window=global_window,
        window=global_window_low,
        min_count=global_min_count, workers=global_workers,
        # epochs=global_epochs,
        epochs=global_epochs,
        seed=global_seed,
    )
    wpa_model_lowdim_gend.save("w2v_wpa_lowdim_gend.model")
    vprint("Low-dim w2v models trained")

    # For the main keywords
    df = get_embedding_df(wpa_model_lowdim_gend, kw_main)

    # Since "slave" and "master" are so close they overlap
    adjust_points = True
    if adjust_points:
        wpa_adjust = {
            # "slave": (0,0.05),
            "master": (0.0, 0.03),
            # "black": (0,0),
            # "white": (0,0),
        }
        df.set_index("token", inplace=True)
        for adj_token, adj_shift in wpa_adjust.items():
            df.loc[adj_token, "x"] = df.loc[adj_token, "x"] + adj_shift[0]
            df.loc[adj_token, "y"] = df.loc[adj_token, "y"] + adj_shift[1]
        df.reset_index(inplace=True)

    # Set x/y lims so there's padding (to avoid labels being cut off)
    pad = 0.5
    xlim = (df['x'].min() - pad, df['x'].max() + pad)
    width = xlim[1] - xlim[0]
    ylim = (df['y'].min() - pad, df['y'].max() + pad)
    height = ylim[1] - ylim[0]
    label_adj = {
        # "white": (-0.075*width,0),
        "black": (-0.075 * width, 0),
        "woman": (-0.09 * width, 0),
        "man": (-0.065 * width, 0),
        "slave": (-0.075 * width, 0),
    }

    # Plot 1: Raw embeddings
    arrow_data = [
        {'pointA': 'white', 'pointB': 'black', 'astyle': '->'},
        {'pointA': 'master', 'pointB': 'slave', 'astyle': '->'},
        {'pointA': 'man', 'pointB': 'woman', 'lstyle': '-.'}
    ]
    gen_plot_svg(df, xvar='x', yvar='y', title="WPA Embeddings",
                 svg_fname="wpa_embeddings.svg",
                 xlim=xlim, ylim=ylim, label_adjustments=label_adj,
                 arrow_data=arrow_data)

    # Export to LaTeX
    df.set_index('token', inplace=True)
    tex_str_standalone = hptutil.custom_latex_export(df, self_contained=True, debug=False)
    tex_standalone_fpath = os.path.join(fig_path, "wpa_embeddings_standalone.tex")
    with open(tex_standalone_fpath, 'w', encoding='utf-8') as outfile:
        outfile.write(tex_str_standalone)
    print(f"Raw embeddings plot saved to {tex_standalone_fpath}")
    # Version for paper
    tex_str_paper = hptutil.custom_latex_export(df, self_contained=False)
    tex_output_fpath = os.path.join(fig_path, "wpa_embeddings.tex")
    with open(tex_output_fpath, 'w', encoding='utf-8') as outfile:
        outfile.write(tex_str_paper)

    # tSNE
    cur_model = wpa_model_lowwin_gend
    tsne_vecs = cur_model.wv.vectors
    tsne_vecs_normed = cur_model.wv.get_normed_vectors()
    # Change this to change whether it uses normed vectors
    tsne_vec_df = pd.DataFrame(tsne_vecs_normed, index=cur_model.wv.index_to_key)
    tsne_N = 1500
    top_n_tokens = [t[0] for t in tcounts[:tsne_N]]
    tsne_topn_df = tsne_vec_df[tsne_vec_df.index.isin(top_n_tokens)].copy()
    vprint("Training tSNE model...")
    tsne_model = TSNE(n_components=2, random_state=global_seed,
                      metric='cosine', square_distances=True,
                      learning_rate=10,
                      perplexity=12.0, n_jobs=8,
                      n_iter=2000)
    X_tsne_topn = tsne_model.fit_transform(tsne_topn_df)
    tsne_df_full = pd.DataFrame(X_tsne_topn, columns=['x', 'y'], index=tsne_topn_df.index)
    vprint("tSNE model trained")

    # Now just the keywords
    kw_remove = ['freedom', 'slavery']
    kw_filtered = [kw for kw in kw_tsne if kw not in kw_remove]
    # kw_add = ['plantation', 'rich', 'poor', 'child']
    kw_add = ['baby', 'child', 'rich', 'poor']
    kw_final = kw_filtered + kw_add
    tsne_df = tsne_df_full[tsne_df_full.index.isin(kw_final)].copy()

    # Rescaling the poor/rich, since they're so far away in the y direction you can't
    # see the angles in the plot
    tsne_df.at['centroid', 'x'] = tsne_df['x'].mean()
    tsne_df.at['centroid', 'y'] = tsne_df['y'].mean()
    dilation_amount = 0.85

    contract_midpoint(tsne_df, 'poor', dilation_amount)
    contract_midpoint(tsne_df, 'rich', dilation_amount)
    # tsne_df.at['rich','y'] = tsne_df.loc['rich', 'y'] / 3
    # And now we can drop the centroid
    tsne_df = tsne_df[tsne_df.index != 'centroid'].copy()
    # Compute averages
    white_master = (tsne_df.loc['white'] + tsne_df.loc['master']) / 2
    black_slave = (tsne_df.loc['black'] + tsne_df.loc['slave']) / 2

    def gen_point(x, y):
        return pd.Series({'x': x, 'y': y})

    tsne_df.at['whitemaster'] = gen_point(white_master['x'], white_master['y'])
    tsne_df.at['blackslave'] = gen_point(black_slave['x'], black_slave['y'])

    # Find where it intersects man->woman
    race_vec = tsne_df.loc['blackslave'] - tsne_df.loc['whitemaster']
    gender_vec = tsne_df.loc['woman'] - tsne_df.loc['man']
    race_slope = race_vec['y'] / race_vec['x']
    gender_slope = gender_vec['y'] / gender_vec['x']
    wm_x = tsne_df.loc['whitemaster', 'x']
    wm_y = tsne_df.loc['whitemaster', 'y']
    m_x = tsne_df.loc['man', 'x']
    m_y = tsne_df.loc['man', 'y']
    intersect_numer = (race_slope * wm_x - gender_slope * m_x) + (m_y - wm_y)
    intersect_denom = race_slope - gender_slope
    intersect_x = intersect_numer / intersect_denom
    intersect_y = gender_slope * (intersect_x - m_x) + m_y
    tsne_df.at['intersect'] = gen_point(intersect_x, intersect_y)
    # Angle at the intersection
    cos_theta = np.dot(race_vec, gender_vec) / (np.linalg.norm(race_vec) * np.linalg.norm(gender_vec))
    vec_angle = np.arccos(cos_theta)
    theta_degrees = np.rad2deg(vec_angle)
    theta_degrees = str(np.round(theta_degrees, 2))

    # Set bounds with padding
    # x
    tsne_xmin = tsne_df['x'].min()
    tsne_xmax = tsne_df['x'].max()
    tsne_xrange = tsne_xmax - tsne_xmin
    tsne_xpad = 0.01 * tsne_xrange
    tsne_xlim = (tsne_xmin - tsne_xpad, tsne_xmax + tsne_xpad)
    # y
    tsne_ymin = tsne_df['y'].min()
    tsne_ymax = tsne_df['y'].max()
    tsne_yrange = tsne_ymax - tsne_ymin
    tsne_ypad = 0.01 * tsne_yrange
    tsne_ylim = (tsne_ymin - tsne_ypad, tsne_ymax + tsne_ypad)

    # tSNE plot
    # Plot tSNE like it's the 2d space
    tsne_arrows = [
        {'pointA': 'white', 'pointB': 'black'},
        {'pointA': 'master', 'pointB': 'slave'},
        {'pointA': 'whitemaster', 'pointB': 'blackslave'},
        {'pointA': 'man', 'pointB': 'woman', 'lstyle': '--'},
    ]
    tsne_title = "Word Embeddings for WPA Slave Narrative Collection, tSNE"
    gen_plot_svg(tsne_df, xvar='x', yvar='y', title=tsne_title,
                 svg_fname="wpa_embeddings_tsne.svg", xlim=tsne_xlim, ylim=tsne_ylim,
                 arrow_data=tsne_arrows)

    # And export to LaTeX
    # To latex
    arrow_data = [
        {'pointA': 'man', 'pointB': 'woman', 'linestyle': '', 'arrowtype': '->'},
        # {'pointA': 'master', 'pointB': 'slave', 'linestyle': 'dashed', 'arrowtype': '->'},
        # {'pointA': 'white', 'pointB': 'black', 'linestyle': 'dashed', 'arrowtype': '->'},
        {'pointA': 'whitemaster', 'pointB': 'blackslave', 'linestyle': '', 'arrowtype': '->'},
        {'pointA': 'whitemaster', 'pointB': 'intersect', 'linestyle': 'dashed', 'arrowtype': '-'},
    ]
    label_data = {
        'slave': {'anchor': 'south', 'xshift': 0.0, 'yshift': 0.0},
        'black': {'anchor': 'south'},
        'master': {'anchor': 'north', 'xshift': 12.0, 'yshift': 0.0},
        'woman': {'anchor': 'south', 'xshift': 2.0, 'yshift': 0.0, 'formatted': '${\\large \\entpgf{woman}}$'},
        'man': {'yshift': -2.0, 'formatted': '${\\large \\entpgf{man}}$'},
        'child': {'anchor': 'south'},
        'whitemaster': {'formatted': '${\\large \\entpgf{white} \\bowtie \\entpgf{master}}$',
                        'xshift': 8.0, 'yshift': 4.0, 'anchor': 'south'},
        'blackslave': {'formatted': '${\\large \\entpgf{black} \\bowtie \\entpgf{slave}}$', 'yshift': 0.0,
                       'anchor': 'south'},
        'intersect': {'draw_point': False, 'formatted': f'\\large $\\theta \\approx {theta_degrees}^\circ$',
                      'anchor': 'north', 'xshift': -29.0},
    }
    axis_options = {
        'xlabel': 'Arbitrary Dimension 1',
        'ylabel': 'Arbitrary Dimension 2',
    }
    # Self-contained version
    tsne_tex_standalone = hptutil.custom_latex_export(tsne_df, label_data=label_data, arrow_data=arrow_data,
                                                      self_contained=True, pad_pct=0.1)
    tex_standalone_fpath = "./tsne_standalone.tex"
    with open(tex_standalone_fpath, 'w', encoding='utf-8') as outfile:
        outfile.write(tsne_tex_standalone)
    print(f"tSNE plot saved to {tex_standalone_fpath}")
    # Version for paper
    tsne_tex = hptutil.custom_latex_export(tsne_df, label_data=label_data, arrow_data=arrow_data,
                                           self_contained=False, pad_pct=0.1)
    tex_output_fpath = os.path.join(fig_path, "wpa_embeddings_tsne.tex")
    with open(tex_output_fpath, 'w', encoding='utf-8') as outfile:
        outfile.write(tsne_tex)

    # And save data to .csv
    tsne_df.to_csv("./wpa_vecs_tsne.csv")
    tsne_df_full.to_csv("./wpa_vecs_tsne_full.csv")

    def normalize_vector(orig_vec):
        return orig_vec / np.sqrt(np.sum(orig_vec ** 2))

    tsne_df['vec'] = tsne_df.apply(lambda r: np.array([r['x'], r['y']]), axis=1)

    def get_coord(raw_vec, normed_unit_vec):
        normalized_vec = normalize_vector(raw_vec)
        return normed_unit_vec * np.dot(normalized_vec, normed_unit_vec)

    def ortho_projection(orig_vec, unit_vec):
        return np.dot(orig_vec, unit_vec) / np.dot(unit_vec, unit_vec)

    # tsne_df['gender_coord'] = tsne_df['vec'].apply(lambda x: get_coord(x, gender_vec_norm))
    # tsne_df['race_coord'] = tsne_df['vec'].apply(lambda x: get_coord(x, race_vec_norm))
    tsne_df['ortho_proj_gender'] = tsne_df['vec'].apply(lambda x: ortho_projection(x, gender_vec))
    tsne_df['ortho_proj_race'] = tsne_df['vec'].apply(lambda x: ortho_projection(x, race_vec))

    # Center at 0
    tsne_df['gender_coord'] = tsne_df['ortho_proj_gender'] - tsne_df['ortho_proj_gender'].mean()
    tsne_df['race_coord'] = tsne_df['ortho_proj_race'] - tsne_df['ortho_proj_race'].mean()

    # tSNE projected plot
    # xvar = 'ortho_proj_race'
    xvar = 'race_coord'
    # yvar = 'ortho_proj_gender'
    yvar = 'gender_coord'
    # Line from white to black
    arrow_data = [
        {'pointA': 'white', 'pointB': 'black'},
        {'pointA': 'master', 'pointB': 'slave'},
        {'pointA': 'whitemaster', 'pointB': 'blackslave'},
        {'pointA': 'man', 'pointB': 'woman', 'lstyle': '--'},
    ]
    gen_plot_svg(tsne_df, xvar=xvar, yvar=yvar, title="tSNE Plot, Projected",
                 svg_fname="wpa_tsne_projected.svg", xlim=None, ylim=None)

    # And to LaTeX
    # Cool. So, just drop the original x/y, and now these coords are the x/y
    tsne_df['x'] = tsne_df['race_coord']
    tsne_df['y'] = tsne_df['gender_coord']

    # LaTeX for this version, where the axes are now meaningful
    # To latex
    arrow_data = [
        {'pointA': 'man', 'pointB': 'woman', 'linestyle': '', 'arrowtype': '->'},
        # {'pointA': 'master', 'pointB': 'slave', 'linestyle': 'dashed', 'arrowtype': '->'},
        # {'pointA': 'white', 'pointB': 'black', 'linestyle': 'dashed', 'arrowtype': '->'},
        {'pointA': 'whitemaster', 'pointB': 'blackslave', 'linestyle': '', 'arrowtype': '->'},
        {'pointA': 'whitemaster', 'pointB': 'intersect', 'linestyle': 'dashed', 'arrowtype': '-'},
    ]
    label_data = {
        'slave': {'anchor': 'south', 'xshift': 0.0, 'yshift': 0.0},
        'black': {'anchor': 'south'},
        'master': {'anchor': 'north', 'xshift': 12.0, 'yshift': 0.0},
        'woman': {'anchor': 'south', 'xshift': 2.0, 'yshift': 0.0, 'formatted': '${\\large \\entpgf{woman}}$'},
        'man': {'yshift': -2.0, 'formatted': '${\\large \\entpgf{man}}$'},
        'child': {'anchor': 'south'},
        'whitemaster': {'formatted': '${\\large \\entpgf{white} \\bowtie \\entpgf{master}}$',
                        'xshift': 8.0, 'yshift': 4.0, 'anchor': 'south'},
        'blackslave': {'formatted': '${\\large \\entpgf{black} \\bowtie \\entpgf{slave}}$', 'yshift': 0.0,
                       'anchor': 'south'},
        'intersect': {'draw_point': False, 'formatted': f'\\large $\\theta \\approx {theta_degrees}^\circ$',
                      'anchor': 'north', 'xshift': -29.0},
    }
    axis_options = {
        'xlabel': 'Race Dimension',
        'ylabel': 'Gender Dimension'
    }
    # Self-contained version
    tsne_tex_standalone = hptutil.custom_latex_export(tsne_df, label_data=label_data, arrow_data=arrow_data,
                                                      self_contained=True, pad_pct=0.1, axis_options=axis_options)
    tex_standalone_fpath = "./tsne_standalone_projected.tex"
    with open(tex_standalone_fpath, 'w', encoding='utf-8') as outfile:
        outfile.write(tsne_tex_standalone)
    print(f"Projected plot written to {tex_standalone_fpath}")
    # Version for paper
    tsne_tex = hptutil.custom_latex_export(tsne_df, label_data=label_data, arrow_data=arrow_data,
                                           self_contained=False, pad_pct=0.1)
    tex_output_fpath = os.path.join(fig_path, "wpa_tsne_projected.tex")
    with open(tex_output_fpath, 'w', encoding='utf-8') as outfile:
        outfile.write(tsne_tex)


if __name__ == "__main__":
    main()

