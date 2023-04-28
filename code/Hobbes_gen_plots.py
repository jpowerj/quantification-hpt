from collections import Counter
import glob
import itertools
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def set_seaborn_style():
    # Idk why we have to call this twice for it to work but... here we are
    sns.set(rc={'figure.figsize':(8,5)})
    sns.set_theme(style="whitegrid")
set_seaborn_style()
import nltk
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import joblib
from tqdm import tqdm
tqdm.pandas()

import hptutil

global_window_high = 21
global_window_low = 9
global_epochs = 80
global_min_count = 10
global_seed = 1949
fig_path = os.path.join("..","figs")

stopwords_en = nltk.corpus.stopwords.words('english')

keywords = [
    #'covenant',
    #'god',
    'authority',
    'christ',
    'civill',
    'common-wealth',
    #'duty',
    #'externall',
    'externall_impediment',
    'externall-impediment',
    'fear',
    'free',
    'government',
    'holy',
    #'impediment',
    #'internall',
    #'judge',
    'justice',
    'king',
    'kingdome',
    'law',
    'liberty',
    'lord',
    #'man',
    'obey',
    'power',
    #'reason',
    'soveraign',
    #'subject'
    #'supreme',
    #'obligation',
    #'right',
    #'nature',
]

join_tokens = [
    ('externall','impediment'),
]

token_clusters = [
    ['christ','jesus'],
    ['externall','external'],
    ['fear','feare','feared','fears','fearing','feares','fearfull','feareth','feare—'],
    ['free','freely','freedome','free-will','freewill'],
    ['god','gods'],
    ['impediment','impediments'],
    ['internall','internal'],
    ['king','kings'],
    ['law','laws','lawes'],
    ['liberty', 'libertie', 'libertatis', 'libertas'],
    ['man','men'],
    ['obey', 'obedience'],
    ['soveraign','soveraignty','soveraigne'],
    ['subject','subjects'],
]
cluster_map = {}
for cur_cluster in token_clusters:
    cluster_main = cur_cluster[0]
    for cur_token in cur_cluster:
        cluster_map[cur_token] = cluster_main
kw_pairs = list(itertools.combinations_with_replacement(keywords, 2))
#kw_pairs


# For the sake of getting individual token counts, we combine into a giant list here
# (But this is just for sanity-checking)
def combine_sent_tokens(sent_tokens):
    tokens_combined = []
    for cur_sent_tokens in sent_tokens:
        tokens_combined.extend(cur_sent_tokens)
    return tokens_combined

def get_token_counts(clean_sent_tokens):
    clean_tokens_combined = combine_sent_tokens(clean_sent_tokens)
    clean_tokens_count = Counter(clean_tokens_combined)
    counts_sorted = sorted(clean_tokens_count.items(), key=lambda x: x[1], reverse=True)
    return counts_sorted


def preprocess_tokens(token_list):
    punct_list = ".,\"':!?;()“”’&[]1234567890"
    def remove_punct(token):
        return "".join([t for t in token if t not in punct_list])
    clean_tokens = [remove_punct(t) for t in token_list]
    en_stopwords = nltk.corpus.stopwords.words('english')
    clean_tokens = [t for t in clean_tokens if t not in en_stopwords]

    #cluster_map
    def collapse_token(orig_token):
        return cluster_map[orig_token] if orig_token in cluster_map else orig_token
    clean_tokens = [collapse_token(t) for t in clean_tokens]
    # Join phrases
    for token_index in range(len(clean_tokens)-1):
        cur_token = clean_tokens[token_index]
        next_token = clean_tokens[token_index+1]
        token_tuple = (cur_token, next_token)
        if cur_token == "externall" or next_token == "externall":
            print(cur_token, next_token)
        if (cur_token, next_token) in join_tokens:
            print(f"Joining {token_tuple}")
            clean_tokens[token_index] = cur_token + "_" + next_token
            clean_tokens[token_index+1] = ""
    # Lastly, remove any blank tokens
    clean_tokens = [t for t in clean_tokens if len(t) > 0]
    return clean_tokens

def load_texts(text_list):
    text_path = os.path.join("..", "corpus_hobbes")
    raw_texts = []
    for text_name in text_list:
        fname = f"{text_name}.txt"
        fpath = os.path.join(text_path, fname)
        with open(fpath, 'r', encoding='utf-8') as infile:
            cur_text = infile.read()
            raw_texts.append(cur_text)
    raw_text = "\n\n\n".join(raw_texts)
    return raw_text

def get_embedding_df(w2v_model, kw_list, tagged=False):
    w2v_rows = []
    emb_keywords = kw_list
    if tagged:
        emb_keywords = [kw+"[m]" for kw in kw_list] + [kw+"[h]" for kw in kw_list]
    for cur_kw in emb_keywords:
        if cur_kw not in w2v_model.wv:
            continue
        cur_vec = w2v_model.wv[cur_kw]
        cur_row = [cur_kw] + list(cur_vec)
        w2v_rows.append(cur_row)
    w2v_df = pd.DataFrame(w2v_rows, columns=['token','x','y'])
    return w2v_df

def process_texts(text_list, fname):
    text_path = os.path.join("..","corpus_hobbes")
    default_texts = ['behemoth', 'de_cive', 'elements', 'leviathan']
    text_str = load_texts(text_list)
    clean_text = text_str.lower()
    sents = nltk.tokenize.sent_tokenize(clean_text, "english")
    sent_tokens = [nltk.tokenize.word_tokenize(sent) for sent in sents]
    clean_sent_tokens = [preprocess_tokens(sent_tokens) for sent_tokens in sent_tokens]
    token_counts = get_token_counts(clean_sent_tokens)
    all_tokens = combine_sent_tokens(clean_sent_tokens)
    print("Training w2v")
    w2v_model = Word2Vec(sentences=clean_sent_tokens, vector_size=2, window=global_window_low,
                         min_count=1, workers=8, epochs=global_epochs, seed=global_seed)
    w2v_model.save(f"w2v_{fname}.model")
    print("w2v complete")
    # For the main keywords
    df = get_embedding_df(w2v_model, keywords)
    # Raw embeddings tex plot
    if df.index.name != "token":
        df.set_index("token", inplace=True)
    latex_str = hptutil.custom_latex_export(df, self_contained=True, debug=False)
    with open(f"{fname}_standalone.tex", 'w', encoding='utf-8') as outfile:
        outfile.write(latex_str)
    # tSNE
    cur_model = w2v_model
    tsne_vecs = cur_model.wv.vectors
    tsne_vecs_normed = cur_model.wv.get_normed_vectors()
    # Change this to change whether it uses normed vectors
    tsne_vec_df = pd.DataFrame(tsne_vecs_normed, index=cur_model.wv.index_to_key)
    tsne_N = 8000
    top_n_tokens = [t[0] for t in token_counts[:tsne_N]]
    tsne_topn_df = tsne_vec_df[tsne_vec_df.index.isin(top_n_tokens)].copy()
    print("Training tSNE")
    tsne = TSNE(n_components=2, random_state=global_seed,
                metric='cosine',
                learning_rate=10,
                perplexity=12.0, n_jobs=8,
                n_iter=2000, square_distances=True)
    X_tsne_topn = tsne.fit_transform(tsne_topn_df)
    print("tSNE complete")
    tsne_df_full = pd.DataFrame(X_tsne_topn, columns=['x', 'y'], index=tsne_topn_df.index)
    tsne_df = tsne_df_full[tsne_df_full.index.isin(keywords)].copy()
    # tex export
    latex_str = hptutil.custom_latex_export(tsne_df, self_contained=True, debug=False, pad_pct=0.2)
    latex_fpath = f"{fname}_tsne_standalone.tex"
    with open(latex_fpath, 'w', encoding='utf-8') as outfile:
        outfile.write(latex_str)
    print(f"Exported to {latex_fpath}")

if __name__ == "__main__":
    if sys.argv[1] == "lev":
        text_list = ['leviathan']
        fname = "lev"
    else:
        text_list = ['de_cive', 'elements', 'behemoth']
        fname = "nonlev"
    process_texts(text_list, fname)
