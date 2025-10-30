# --------------------------------------------------------------
# style_analyzer_gen3.py  (FINAL – NO KeyError, FULLY WORKING)
# --------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import hdbscan
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2

from src.constants import Columns
from loguru import logger


# ------------------------------------------------------------------
# 1. Load enriched + BRUTE-FORCE COLUMN NAMES
# ------------------------------------------------------------------
def load_enriched() -> pd.DataFrame:
    from pathlib import Path
    import re
    from datetime import datetime

    processed_dir = Path("data") / "processed"
    pattern = re.compile(r"whatsapp_all_enriched-(\d{8}-\d{6})\.csv")

    enriched_files = [f for f in processed_dir.glob("whatsapp_all_enriched-*.csv") if pattern.match(f.name)]
    if not enriched_files:
        raise FileNotFoundError("No enriched CSV found.")

    latest = max(enriched_files,
                 key=lambda f: datetime.strptime(pattern.match(f.name).group(1), "%Y%m%d-%H%M%S"))

    logger.info(f"Loading enriched data: {latest}")
    df = pd.read_csv(latest, parse_dates=[Columns.TIMESTAMP.value])

    # ---- BRUTE-FORCE RENAME ANY VARIANT ----
    rename_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower == 'author':
            rename_map[col] = Columns.AUTHOR.value
        elif lower == 'year':
            rename_map[col] = Columns.YEAR.value
        elif lower == 'whatsapp_group':
            rename_map[col] = Columns.WHATSAPP_GROUP.value

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Renamed columns: {rename_map}")

    # Verify required
    required = [Columns.AUTHOR.value, Columns.YEAR.value, Columns.WHATSAPP_GROUP.value]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CRITICAL: Missing columns after rename: {missing}")

    if 'message_cleaned' not in df.columns:
        df['message_cleaned'] = ""

    return df


# ------------------------------------------------------------------
# 2. Choose aggregation level
# ------------------------------------------------------------------
def choose_group_cols(by_year: bool = False) -> list[str]:
    cols = [Columns.AUTHOR.value]
    if by_year:
        cols.append(Columns.YEAR.value)
    return cols


# ------------------------------------------------------------------
# 3. Core aggregation – PRESERVE whatsapp_group
# ------------------------------------------------------------------
def aggregate_basic(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    # Include whatsapp_group in groupby to preserve it
    full_group_cols = group_cols + [Columns.WHATSAPP_GROUP.value]

    agg_dict = {
        'msg_count': ('message_cleaned', 'count'),
        'words_total': (Columns.NUMBER_OF_WORDS.value, 'sum'),
        'emojis_total': (Columns.NUMBER_OF_EMOJIS.value, 'sum'),
        'punct_total': (Columns.NUMBER_OF_PUNCTUATIONS.value, 'sum'),
        'caps_total': (Columns.NUMBER_OF_CAPITALS.value, 'sum'),
        'pics_total': (Columns.NUMBER_OF_PICTURES_VIDEOS.value, 'sum'),
        'links_total': (Columns.HAS_LINK.value, 'sum'),
        'attachments_total': (Columns.HAS_ATTACHMENT.value, 'sum'),
        'mean_response': (Columns.RESPONSE_TIME.value, 'mean'),
        'std_response': (Columns.RESPONSE_TIME.value, 'std'),
        'pct_ends_emoji': (Columns.ENDS_WITH_EMOJI.value, 'mean'),
        'pct_ends_punct': (Columns.ENDS_WITH_PUNCTUATION.value, 'mean'),
        'pct_has_capitals': (Columns.HAS_CAPITALS.value, 'mean'),
        'avg_word_len': (Columns.AVG_WORD_LENGTH.value, 'mean'),
        'mean_chat_len': (Columns.LENGTH_CHAT.value, 'mean'),
        'mean_caps_ratio': (Columns.CAPITALIZED_WORDS_RATIO.value, 'mean'),
        'pct_starts_capital': (Columns.STARTS_WITH_CAPITAL.value, 'mean'),
        'mean_pct_emojis': (Columns.PCT_EMOJIS.value, 'mean'),
        'pct_starts_emoji': (Columns.STARTS_WITH_EMOJI.value, 'mean'),
        'mean_pct_punct': (Columns.PCT_PUNCTUATIONS.value, 'mean'),
        'pct_has_question': (Columns.HAS_QUESTION_MARK.value, 'mean'),
        'pct_ends_question': (Columns.ENDS_WITH_QUESTION_MARK.value, 'mean'),
        'numbers_total': (Columns.NUMBER_OF_NUMBERS.value, 'sum'),
        'pct_was_deleted': (Columns.WAS_DELETED.value, 'mean'),
    }

    agg = df.groupby(full_group_cols).agg(**agg_dict).reset_index()

    # pct_replies
    prev = df[Columns.PREVIOUS_AUTHOR.value]
    curr = df[Columns.AUTHOR.value]
    reply_mask = prev.notna() & (prev != curr)
    group_key = df[full_group_cols].apply(tuple, axis=1)
    reply_pct = reply_mask.groupby(group_key).mean()
    reply_df = reply_pct.reset_index(name='pct_replies')
    reply_df[full_group_cols] = pd.DataFrame(reply_df['index'].tolist(), columns=full_group_cols)
    reply_df = reply_df.drop(columns='index')
    agg = agg.merge(reply_df, on=full_group_cols, how='left').fillna(0)

    # Burst counts
    burst_defs = [
        (Columns.LIST_OF_CONNECTED_PUNCTUATIONS.value, {
            '!!!': 'exclamation_3', '!!!!': 'exclamation_4plus', '!!': 'exclamation_2',
            '...': 'ellipses_3', '....': 'ellipses_4plus',
            '??': 'question_2', '???': 'question_3plus',
            '?!': 'surprise_qe', '!?': 'surprise_eq',
        }),
        (Columns.LIST_OF_CONNECTED_EMOJIS.value, {
            '😂😂': 'laugh_2', '😂😂😂': 'laugh_3', '😂😂😂😂': 'laugh_4plus',
            '🤣🤣': 'laugh_2', '🤣🤣🤣': 'laugh_3plus',
            '😄😄': 'happy_2',
            '👍👍': 'thumbs_2', '👍🏻👍🏻': 'thumbs_2', '👍🏼👍🏼': 'thumbs_2',
            '❤️': 'heart_single',
        }),
        (Columns.LIST_OF_CONNECTED_CAPITALS.value, {
            'OK': 'ok_burst', 'ZS': 'zs_burst', 'ETA': 'eta_burst',
        })
    ]

    for list_col, mapping in burst_defs:
        if list_col not in df.columns:
            continue
        exploded = df[full_group_cols + [list_col]].explode(list_col)
        exploded = exploded[exploded[list_col].notna()]
        if not mapping:
            continue
        mask = exploded[list_col].isin(mapping.keys())
        filtered = exploded[mask]
        if filtered.empty:
            continue
        vc = filtered.groupby(full_group_cols)[list_col].value_counts().unstack(fill_value=0)
        vc = vc.rename(columns=mapping)
        counts = vc.reindex(columns=mapping.values(), fill_value=0).reset_index()
        agg = agg.merge(counts, on=full_group_cols, how='left').fillna(0)

    return agg


# ------------------------------------------------------------------
# 4. Build style features
# ------------------------------------------------------------------
def build_style_features(agg: pd.DataFrame) -> tuple:
    mc = agg['msg_count'].replace(0, np.nan)

    agg['words_per_msg'] = agg['words_total'] / mc
    agg['emoji_per_msg'] = agg['emojis_total'] / mc
    agg['punct_per_msg'] = agg['punct_total'] / mc
    agg['caps_per_word'] = agg['caps_total'] / agg['words_total'].replace(0, np.nan)
    agg['media_per_msg'] = (agg['pics_total'] + agg['attachments_total']) / mc

    agg['exclamation_per_msg'] = (agg.get('exclamation_2', 0) +
                                  agg.get('exclamation_3', 0) +
                                  agg.get('exclamation_4plus', 0)) / mc
    agg['ellipses_per_msg'] = (agg.get('ellipses_3', 0) +
                               agg.get('ellipses_4plus', 0)) / mc
    agg['question_per_msg'] = (agg.get('question_2', 0) +
                               agg.get('question_3plus', 0)) / mc
    agg['surprise_per_msg'] = (agg.get('surprise_qe', 0) +
                               agg.get('surprise_eq', 0)) / mc
    agg['laugh_burst_per_msg'] = (agg.get('laugh_2', 0) +
                                  agg.get('laugh_3', 0) +
                                  agg.get('laugh_3plus', 0) +
                                  agg.get('laugh_4plus', 0)) / mc
    agg['thumbs_burst_per_msg'] = agg.get('thumbs_2', 0) / mc
    agg['happy_burst_per_msg'] = agg.get('happy_2', 0) / mc
    agg['shortcode_burst_per_msg'] = (agg.get('ok_burst', 0) +
                                      agg.get('zs_burst', 0) +
                                      agg.get('eta_burst', 0)) / mc
    agg['numbers_per_msg'] = agg['numbers_total'] / mc

    style_cols = [
        'words_per_msg', 'emoji_per_msg', 'punct_per_msg', 'caps_per_word',
        'media_per_msg', 'exclamation_per_msg', 'ellipses_per_msg',
        'pct_ends_emoji', 'pct_ends_punct', 'pct_has_capitals',
        'mean_response', 'pct_replies',
        'avg_word_len', 'mean_chat_len', 'mean_caps_ratio', 'pct_starts_capital',
        'mean_pct_emojis', 'pct_starts_emoji',
        'mean_pct_punct', 'pct_has_question', 'pct_ends_question',
        'numbers_per_msg', 'pct_was_deleted',
        'question_per_msg', 'surprise_per_msg',
        'laugh_burst_per_msg', 'thumbs_burst_per_msg', 'happy_burst_per_msg',
        'shortcode_burst_per_msg'
    ]

    X = agg[style_cols].fillna(0)
    agg = agg.join(X.add_prefix('style_'))
    return agg, X, style_cols


# ------------------------------------------------------------------
# 5. Reduce dimensions
# ------------------------------------------------------------------
def reduce_dimensions(X: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    pca2d = pca.fit_transform(X_scaled)

    tsne = TSNE(n_components=2,
                perplexity=min(15, len(X) // 3),
                random_state=42,
                init='pca',
                learning_rate='auto')
    tsne2d = tsne.fit_transform(X_scaled)

    red = pd.DataFrame({
        'pca_x': pca2d[:, 0], 'pca_y': pca2d[:, 1],
        'tsne_x': tsne2d[:, 0], 'tsne_y': tsne2d[:, 1],
    })
    return red


# ------------------------------------------------------------------
# 6. HDBSCAN clustering
# ------------------------------------------------------------------
def cluster_styles(X_scaled: np.ndarray, agg: pd.DataFrame) -> pd.DataFrame:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
    labels = clusterer.fit_predict(X_scaled)
    agg['style_cluster'] = labels

    name_map = {
        0: "Expressive / Playful",
        1: "Intense / Shouter",
        2: "Reflective / Visual",
        3: "Formal / Structured",
        -1: "Unique / Outlier"
    }
    agg['style_name'] = agg['style_cluster'].map(name_map).fillna("Unique")
    return agg


# ------------------------------------------------------------------
# 7. Plot helpers
# ------------------------------------------------------------------
def hex_to_rgba(hex_color, alpha=1.0):
    h = hex_color.lstrip('#')
    return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})'

def get_ellipse_points(mean_x, mean_y, width, height, angle, num_points=100):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = mean_x + (width/2)*np.cos(theta)*np.cos(np.radians(angle)) - (height/2)*np.sin(theta)*np.sin(np.radians(angle))
    y = mean_y + (width/2)*np.cos(theta)*np.sin(np.radians(angle)) + (height/2)*np.sin(theta)*np.cos(np.radians(angle))
    return x, y


# ------------------------------------------------------------------
# 8. Individual t-SNE
# ------------------------------------------------------------------
def plot_tsne_individual(agg: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    agg['label'] = agg[Columns.AUTHOR.value]
    if Columns.YEAR.value in agg.columns:
        agg['label'] = agg[Columns.AUTHOR.value] + "'" + agg[Columns.YEAR.value].astype(str).str[2:]

    fig = px.scatter(
        agg,
        x='tsne_x', y='tsne_y',
        color=Columns.AUTHOR.value,
        size='msg_count',
        hover_data={
            'style_words_per_msg': ':.1f',
            'style_emoji_per_msg': ':.3f',
            'style_exclamation_per_msg': ':.3f',
            'style_laugh_burst_per_msg': ':.3f',
            'style_avg_word_len': ':.2f',
            'style_pct_has_question': ':.2f',
            'style_name': True,
            'msg_count': True
        },
        title="Linguistic Style Clusters (t-SNE) – Individual Authors – Conf 75%",
        width=950, height=650
    )
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color='black')))

    colors = px.colors.qualitative.Plotly
    author_color_map = {auth: colors[i % len(colors)] for i, auth in enumerate(agg[Columns.AUTHOR.value].unique())}

    for author in agg[Columns.AUTHOR.value].unique():
        sub = agg[agg[Columns.AUTHOR.value] == author]
        if len(sub) < 2: continue
        x, y = sub['tsne_x'].values, sub['tsne_y'].values
        cov = np.cov(x, y)
        mean_x, mean_y = np.mean(x), np.mean(y)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        chi = np.sqrt(chi2.ppf(0.75, 2))
        width, height = 2 * lambda_[0] * chi, 2 * lambda_[1] * chi
        angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
        ell_x, ell_y = get_ellipse_points(mean_x, mean_y, width, height, angle)
        color = author_color_map[author]

        fig.add_trace(go.Scatter(x=ell_x, y=ell_y, mode='lines', fill='toself',
                                 fillcolor=hex_to_rgba(color, 0.2), line=dict(color=color, width=2),
                                 name=f'{author} 75%', showlegend=False, hoverinfo='skip'))
        fig.add_annotation(x=mean_x, y=mean_y, text=author, showarrow=False,
                           font=dict(size=12, color=color), bgcolor="white",
                           bordercolor=color, borderwidth=1, borderpad=4, opacity=0.9)

    path = out_dir / "style_tsne_individual.html"
    fig.write_html(str(path))
    logger.success(f"Individual t-SNE → {path}")
    fig.show()


# ------------------------------------------------------------------
# 9. Group-level t-SNE
# ------------------------------------------------------------------
def plot_tsne_group(agg: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- CREATE 5 GROUPS: AvT is a separate group ---
    def assign_group(row):
        if row[Columns.AUTHOR.value] == "Anthony van Tilburg":
            return "AvT"  # AvT is its own group
        return row[Columns.WHATSAPP_GROUP.value]  # maap, dac, golfmaten, tillies

    agg["plot_group"] = agg.apply(assign_group, axis=1)

    # --- 5 DISTINCT COLORS ---
    group_colors = {
        "maap": "#1f77b4",      # blue
        "dac": "#ff7f0e",       # orange
        "golfmaten": "#2ca02c", # green
        "tillies": "#808080",   # gray
        "AvT": "#d62728"        # RED
    }

    fig = px.scatter(
        agg,
        x="tsne_x", y="tsne_y",
        color="plot_group",
        size="msg_count",
        color_discrete_map=group_colors,
        hover_data={
            "style_words_per_msg": ":.1f",
            "style_exclamation_per_msg": ":.3f",
            "style_laugh_burst_per_msg": ":.3f",
            "msg_count": True,
            Columns.AUTHOR.value: True,
            Columns.WHATSAPP_GROUP.value: True
        },
        title="Linguistic Style Clusters (t-SNE) – 5 Groups (AvT Isolated) – Conf 50%",
        width=950, height=650
    )
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color='black')))

    # 50% confidence ellipses for each of the 5 groups
    for grp in ["maap", "dac", "golfmaten", "tillies", "AvT"]:
        sub = agg[agg["plot_group"] == grp]
        if len(sub) < 2:
            continue
        x, y = sub["tsne_x"].values, sub["tsne_y"].values
        cov = np.cov(x, y)
        mean_x, mean_y = np.mean(x), np.mean(y)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        chi = np.sqrt(chi2.ppf(0.50, 2))
        width, height = 2 * lambda_[0] * chi, 2 * lambda_[1] * chi
        angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
        ell_x, ell_y = get_ellipse_points(mean_x, mean_y, width, height, angle)
        color = group_colors.get(grp, "#333333")

        fig.add_trace(go.Scatter(x=ell_x, y=ell_y, mode='lines', fill='toself',
                                 fillcolor=hex_to_rgba(color, 0.25), line=dict(color=color, width=2),
                                 name=f'{grp} 50%', showlegend=False, hoverinfo='skip'))
        fig.add_annotation(x=mean_x, y=mean_y, text=grp, showarrow=False,
                           font=dict(size=14, color='white', family="Arial Black"),
                           bgcolor=color, borderpad=6, opacity=0.9)

    path = out_dir / "style_tsne_group.html"
    fig.write_html(str(path))
    logger.success(f"5-Group t-SNE (AvT isolated) → {path}")
    fig.show()


# ------------------------------------------------------------------
# 10. Full pipeline
# ------------------------------------------------------------------
def run_style_analysis(
    by_year: bool = True,
    do_clustering: bool = True,
    by_group: bool = False,
    out_dir: Path = Path("style_output")
):
    df = load_enriched()
    group_cols = choose_group_cols(by_year=by_year)
    agg = aggregate_basic(df, group_cols)
    agg, X, style_cols = build_style_features(agg)
    red = reduce_dimensions(X)
    agg = pd.concat([agg, red], axis=1)

    if do_clustering:
        X_scaled = StandardScaler().fit_transform(X)
        agg = cluster_styles(X_scaled, agg)

    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / "style_summary.csv", index=False)

    if by_group:
        plot_tsne_group(agg, out_dir)
    else:
        plot_tsne_individual(agg, out_dir)

    return agg


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
if __name__ == "__main__":
    agg = run_style_analysis(by_year=True, by_group=True)  # Group view
    # agg = run_style_analysis(by_year=True, by_group=False)  # Individual view