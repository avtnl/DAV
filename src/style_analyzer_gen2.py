# --------------------------------------------------------------
# style_analyzer.py  (PATCHED – uses only pre-computed columns)
# --------------------------------------------------------------
import pandas as pd
import numpy as np
import re
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
# 1. Load enriched (tolerant to missing message_cleaned)
# ------------------------------------------------------------------
def load_enriched() -> pd.DataFrame:
    from pathlib import Path
    import re
    from datetime import datetime

    processed_dir = Path("data") / "processed"
    pattern = re.compile(r"whatsapp_all_enriched-(\d{8}-\d{6})\.csv")

    enriched_files = [f for f in processed_dir.glob("whatsapp_all_enriched-*.csv") if pattern.match(f.name)]
    if not enriched_files:
        raise FileNotFoundError("No enriched CSV found. Run run_enrich.py first.")

    latest = max(enriched_files,
                 key=lambda f: datetime.strptime(pattern.match(f.name).group(1), "%Y%m%d-%H%M%S"))

    logger.info(f"Loading enriched data: {latest}")
    df = pd.read_csv(latest, parse_dates=[Columns.TIMESTAMP.value])

    # ---- OPTIONAL: fallback if message_cleaned is missing (only for msg_count) ----
    if 'message_cleaned' not in df.columns:
        logger.warning("message_cleaned missing – using row count as msg_count")
        df['message_cleaned'] = ""          # placeholder
    # ---------------------------------------------------------------------------

    required = [
        'ends_with_emoji', 'ends_with_punctuation', 'has_attachment',
        'has_capitals', 'number_of_capitals', 'number_of_pictures_videos',
        'number_of_punctuations', 'number_of_words', 'previous_author', 'response_time'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {latest}: {missing}")

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
# 3. Core aggregation – now includes categories 1-4
# ------------------------------------------------------------------
def aggregate_basic(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    # --------------------------------------------------------------
    # Basic counts (still need a message counter)
    # --------------------------------------------------------------
    agg_dict = {
        # message count – works even if message_cleaned is empty
        'msg_count': ('message_cleaned', 'count'),

        # totals
        'words_total': (Columns.NUMBER_OF_WORDS.value, 'sum'),
        'emojis_total': (Columns.NUMBER_OF_EMOJIS.value, 'sum'),
        'punct_total': (Columns.NUMBER_OF_PUNCTUATIONS.value, 'sum'),
        'caps_total': (Columns.NUMBER_OF_CAPITALS.value, 'sum'),
        'pics_total': (Columns.NUMBER_OF_PICTURES_VIDEOS.value, 'sum'),
        'links_total': (Columns.HAS_LINK.value, 'sum'),
        'attachments_total': (Columns.HAS_ATTACHMENT.value, 'sum'),

        # response time
        'mean_response': (Columns.RESPONSE_TIME.value, 'mean'),
        'std_response': (Columns.RESPONSE_TIME.value, 'std'),

        # percentages (boolean → mean)
        'pct_ends_emoji': (Columns.ENDS_WITH_EMOJI.value, 'mean'),
        'pct_ends_punct': (Columns.ENDS_WITH_PUNCTUATION.value, 'mean'),
        'pct_has_capitals': (Columns.HAS_CAPITALS.value, 'mean'),

        # ---------- NEW: Text Complexity ----------
        'avg_word_len': (Columns.AVG_WORD_LENGTH.value, 'mean'),
        'mean_chat_len': (Columns.LENGTH_CHAT.value, 'mean'),
        'mean_caps_ratio': (Columns.CAPITALIZED_WORDS_RATIO.value, 'mean'),
        'pct_starts_capital': (Columns.STARTS_WITH_CAPITAL.value, 'mean'),

        # ---------- NEW: Emoji ----------
        'mean_pct_emojis': (Columns.PCT_EMOJIS.value, 'mean'),
        'pct_starts_emoji': (Columns.STARTS_WITH_EMOJI.value, 'mean'),

        # ---------- NEW: Punctuation ----------
        'mean_pct_punct': (Columns.PCT_PUNCTUATIONS.value, 'mean'),
        'pct_has_question': (Columns.HAS_QUESTION_MARK.value, 'mean'),
        'pct_ends_question': (Columns.ENDS_WITH_QUESTION_MARK.value, 'mean'),

        # ---------- NEW: Numbers & Attachments ----------
        'numbers_total': (Columns.NUMBER_OF_NUMBERS.value, 'sum'),
        'pct_was_deleted': (Columns.WAS_DELETED.value, 'mean'),
    }

    agg = df.groupby(group_cols).agg(**agg_dict).reset_index()

    # --------------------------------------------------------------
    # pct_replies (unchanged)
    # --------------------------------------------------------------
    def compute_pct_replies(g):
        return (g[Columns.PREVIOUS_AUTHOR.value].notna() &
                (g[Columns.PREVIOUS_AUTHOR.value] != g[Columns.AUTHOR.value])).mean()

    replies = df.groupby(group_cols).apply(compute_pct_replies).reset_index(name='pct_replies')
    agg = agg.merge(replies, on=group_cols, how='left').fillna(0)

    # --------------------------------------------------------------
    # Burst counts from pre-computed list columns
    # --------------------------------------------------------------
    burst_defs = [
        # (list_column, pattern → burst_column)
        (Columns.LIST_OF_CONNECTED_PUNCTUATIONS.value,
         {'!!!': 'exclamation_bursts',
          '...': 'ellipses',
          '?!': 'question_exclamation',
          '!?': 'question_exclamation',
          '--': 'dash_style',
          '——': 'dash_style',
          '–': 'dash_style',
          '—': 'dash_style'}),
        (Columns.LIST_OF_CONNECTED_EMOJIS.value,
         {None: 'emoji_bursts'}),          # any sequence → count
        (Columns.LIST_OF_CONNECTED_CAPITALS.value,
         {None: 'caps_bursts'})            # any all-caps burst
    ]

    for list_col, mapping in burst_defs:
        if list_col not in df.columns:
            continue
        exploded = df[group_cols + [list_col]].explode(list_col)
        exploded = exploded[exploded[list_col].notna()]

        if mapping is None:  # count any non-null entry
            counts = exploded.groupby(group_cols).size().reset_index(name='burst_cnt')
            burst_name = list_col.split('_')[-1] + '_bursts'
            counts = counts.rename(columns={'burst_cnt': burst_name})
        else:
            vc = exploded.groupby(group_cols)[list_col].value_counts().unstack(fill_value=0)
            # keep only the patterns we care about
            keep = {k: v for k, v in mapping.items() if k in vc.columns}
            if not keep:
                continue
            vc = vc.rename(columns=keep)
            counts = vc.reindex(columns=keep.values(), fill_value=0).reset_index()

        agg = agg.merge(counts, on=group_cols, how='left').fillna(0)

    return agg


# ------------------------------------------------------------------
# 4. Build the final style-feature matrix
# ------------------------------------------------------------------
def build_style_features(agg: pd.DataFrame) -> tuple:
    mc = agg['msg_count'].replace(0, np.nan)

    # ---- existing ----
    agg['words_per_msg'] = agg['words_total'] / mc
    agg['emoji_per_msg'] = agg['emojis_total'] / mc
    agg['punct_per_msg'] = agg['punct_total'] / mc
    agg['caps_per_word'] = agg['caps_total'] / agg['words_total'].replace(0, np.nan)
    agg['media_per_msg'] = (agg['pics_total'] + agg['attachments_total']) / mc
    agg['exclamation_per_msg'] = agg.get('exclamation_bursts', 0) / mc
    agg['ellipses_per_msg'] = agg.get('ellipses', 0) / mc

    # ---- NEW derived features (categories 1-4) ----
    agg['numbers_per_msg'] = agg['numbers_total'] / mc
    agg['question_excl_per_msg'] = agg.get('question_exclamation', 0) / mc
    agg['dash_per_msg'] = agg.get('dash_style', 0) / mc
    agg['emoji_burst_per_msg'] = agg.get('emojis_bursts', 0) / mc
    agg['caps_burst_per_msg'] = agg.get('capitals_bursts', 0) / mc

    style_cols = [
        # existing
        'words_per_msg', 'emoji_per_msg', 'punct_per_msg', 'caps_per_word',
        'media_per_msg', 'exclamation_per_msg', 'ellipses_per_msg',
        'pct_ends_emoji', 'pct_ends_punct', 'pct_has_capitals',
        'mean_response', 'pct_replies',

        # ---- NEW ----
        # Text Complexity
        'avg_word_len', 'mean_chat_len', 'mean_caps_ratio', 'pct_starts_capital',
        # Emoji
        'mean_pct_emojis', 'pct_starts_emoji',
        # Punctuation & Questions
        'mean_pct_punct', 'pct_has_question', 'pct_ends_question',
        # Numbers & Attachments
        'numbers_per_msg', 'pct_was_deleted',
        # Burst rates
        'question_excl_per_msg', 'dash_per_msg',
        'emoji_burst_per_msg', 'caps_burst_per_msg'
    ]

    X = agg[style_cols].fillna(0)
    agg = agg.join(X.add_prefix('style_'))
    return agg, X, style_cols


# ------------------------------------------------------------------
# 5. Dimensionality reduction (unchanged)
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

    umap2d = umap.UMAP(n_neighbors=max(2, min(5, len(X) // 4)),
                       min_dist=0.3,
                       random_state=42).fit_transform(X_scaled)

    red = pd.DataFrame({
        'pca_x': pca2d[:, 0], 'pca_y': pca2d[:, 1],
        'tsne_x': tsne2d[:, 0], 'tsne_y': tsne2d[:, 1],
        'umap_x': umap2d[:, 0], 'umap_y': umap2d[:, 1],
    })
    return red


# ------------------------------------------------------------------
# 6. (Optional) HDBSCAN clustering
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
# 8c. t-SNE with 75% confidence ellipses + author label
# ------------------------------------------------------------------
def plot_tsne(agg: pd.DataFrame, out_dir: Path):
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
            'style_emoji_per_msg': ':.3f',
            'style_caps_per_word': ':.3f',
            'style_exclamation_per_msg': ':.3f',
            'style_ellipses_per_msg': ':.3f',
            'style_words_per_msg': ':.1f',
            'style_name': True,
            'msg_count': True,
            # a few new ones for quick inspection
            'style_avg_word_len': ':.2f',
            'style_pct_has_question': ':.2f',
            'style_emoji_burst_per_msg': ':.3f'
        },
        title="Linguistic Style Clusters (t-SNE) - Conf 75%",
        labels={'tsne_x': 't-SNE Dimension 1', 'tsne_y': 't-SNE Dimension 2'},
        width=950, height=650
    )
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color='black')))

    # ---------- 75% confidence ellipses per author ----------
    unique_authors = agg[Columns.AUTHOR.value].unique()
    colors = px.colors.qualitative.Plotly
    author_color_map = {auth: colors[i % len(colors)] for i, auth in enumerate(unique_authors)}

    for author in unique_authors:
        sub = agg[agg[Columns.AUTHOR.value] == author]
        if len(sub) < 2:
            continue

        x = sub['tsne_x'].values
        y = sub['tsne_y'].values
        cov = np.cov(x, y)
        mean_x, mean_y = np.mean(x), np.mean(y)

        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        chi = np.sqrt(chi2.ppf(0.75, 2))
        width = 2 * lambda_[0] * chi
        height = 2 * lambda_[1] * chi
        angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))

        ell_x, ell_y = get_ellipse_points(mean_x, mean_y, width, height, angle)
        color = author_color_map[author]

        fig.add_trace(go.Scatter(
            x=ell_x, y=ell_y,
            mode='lines',
            fill='toself',
            fillcolor=hex_to_rgba(color, 0.2),
            line=dict(color=color, width=2),
            name=f'{author} 75% ellipse',
            showlegend=False,
            hoverinfo='skip'
        ))

        # author name at ellipse centre
        fig.add_annotation(
            x=mean_x, y=mean_y,
            text=author,
            showarrow=False,
            font=dict(size=12, color=color),
            bgcolor="white",
            bordercolor=color,
            borderwidth=1,
            borderpad=4,
            opacity=0.9
        )

    html_path = out_dir / "style_tsne.html"
    fig.write_html(str(html_path))
    logger.success(f"t-SNE plot with 75% confidence ellipses → {html_path}")
    fig.show()


# ------------------------------------------------------------------
# 9. Full pipeline
# ------------------------------------------------------------------
def run_style_analysis(
    by_year: bool = True,
    do_clustering: bool = True,
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
    summary_path = out_dir / "style_summary.csv"
    agg.to_csv(summary_path, index=False)
    logger.success(f"Style summary → {summary_path}")

    plot_tsne(agg, out_dir)

    return agg


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    agg = run_style_analysis(
        by_year=True,
        do_clustering=True,
        out_dir=Path("style_output")
    )