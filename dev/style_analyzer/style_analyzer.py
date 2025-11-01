# --------------------------------------------------------------
# style_analyzer.py
# --------------------------------------------------------------
import re
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from src.constants import Columns  # <-- your enum with column names


# ------------------------------------------------------------------
# 1. Load the enriched DataFrame (auto-detect newest enriched file)
# ------------------------------------------------------------------
def load_enriched() -> pd.DataFrame:
    import re
    from datetime import datetime
    from pathlib import Path

    processed_dir = Path("data") / "processed"
    pattern = re.compile(r"whatsapp_all_enriched-(\d{8}-\d{6})\.csv")

    enriched_files = [
        f for f in processed_dir.glob("whatsapp_all_enriched-*.csv") if pattern.match(f.name)
    ]

    if not enriched_files:
        raise FileNotFoundError("No enriched CSV found. Run run_enrich.py first.")

    # Pick the newest by timestamp in filename
    latest = max(
        enriched_files,
        key=lambda f: datetime.strptime(pattern.match(f.name).group(1), "%Y%m%d-%H%M%S"),
    )

    logger.info(f"Loading enriched data: {latest}")
    df = pd.read_csv(latest, parse_dates=[Columns.TIMESTAMP.value])

    # Verify required columns
    required = [
        "ends_with_emoji",
        "ends_with_punctuation",
        "has_attachment",
        "has_capitals",
        "number_of_capitals",
        "number_of_pictures_videos",
        "number_of_punctuations",
        "number_of_words",
        "previous_author",
        "response_time",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {latest}: {missing}")

    return df


# ------------------------------------------------------------------
# 2. Choose aggregation level
# ------------------------------------------------------------------
def choose_group_cols(by_year: bool = False) -> list[str]:
    """
    Return the columns that define a “participant”.
    * False → ['author']
    * True  → ['author', 'year']
    """
    cols = [Columns.AUTHOR.value]
    if by_year:
        cols.append(Columns.YEAR.value)
    return cols


# ------------------------------------------------------------------
# 3. Core aggregation (counts + means)
# ------------------------------------------------------------------
def aggregate_basic(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols)
        .agg(
            msg_count=("message_cleaned", "count"),
            words_total=(Columns.NUMBER_OF_WORDS.value, "sum"),
            emojis_total=(Columns.NUMBER_OF_EMOJIS.value, "sum"),
            punct_total=(Columns.NUMBER_OF_PUNCTUATIONS.value, "sum"),
            caps_total=(Columns.NUMBER_OF_CAPITALS.value, "sum"),
            pics_total=(Columns.NUMBER_OF_PICTURES_VIDEOS.value, "sum"),
            links_total=(Columns.HAS_LINK.value, "sum"),
            attachments_total=(Columns.HAS_ATTACHMENT.value, "sum"),
            mean_response=(Columns.RESPONSE_TIME.value, "mean"),
            std_response=(Columns.RESPONSE_TIME.value, "std"),
            pct_ends_emoji=(Columns.ENDS_WITH_EMOJI.value, "mean"),
            pct_ends_punct=(Columns.ENDS_WITH_PUNCTUATION.value, "mean"),
            pct_has_capitals=(Columns.HAS_CAPITALS.value, "mean"),
            # % of messages that are replies (previous author exists AND differs)
            pct_replies=(
                Columns.PREVIOUS_AUTHOR.value,
                lambda s: (s.notna() & (s.shift() != s)).mean() if len(s) > 1 else 0.0,
            ),
        )
        .reset_index()
    )
    return agg


# ------------------------------------------------------------------
# 4. Punctuation-pattern counters (!!!, ..., ?!, ––)
# ------------------------------------------------------------------
def count_punct_patterns(series: pd.Series) -> pd.Series:
    """Return a Series with 4 pattern counts for a group of raw messages."""
    txt = " ".join(series.astype(str))

    exclamation_bursts = len(re.findall(r"!{2,}", txt))
    question_exclamation = len(re.findall(r"\?[!]|![?]", txt))
    ellipses = len(re.findall(r"\.{3,}", txt))
    dash_style = len(re.findall(r"[-–—]{2,}", txt))

    return pd.Series(
        {
            "exclamation_bursts": exclamation_bursts,
            "question_exclamation": question_exclamation,
            "ellipses": ellipses,
            "dash_style": dash_style,
        }
    )


def add_punct_patterns(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    def count_for_group(g):
        return count_punct_patterns(g["message_cleaned"])

    punct = df.groupby(group_cols).apply(count_for_group).reset_index()
    return punct


# ------------------------------------------------------------------
# 5. Build the final style-feature matrix
# ------------------------------------------------------------------
def build_style_features(agg: pd.DataFrame) -> tuple:
    agg["words_per_msg"] = agg["words_total"] / agg["msg_count"]
    agg["emoji_per_msg"] = agg["emojis_total"] / agg["msg_count"]
    agg["punct_per_msg"] = agg["punct_total"] / agg["msg_count"]
    agg["caps_per_word"] = agg["caps_total"] / agg["words_total"].replace(0, np.nan)
    agg["media_per_msg"] = (agg["pics_total"] + agg["attachments_total"]) / agg["msg_count"]
    agg["exclamation_per_msg"] = agg["exclamation_bursts"] / agg["msg_count"]
    agg["ellipses_per_msg"] = agg["ellipses"] / agg["msg_count"]

    style_cols = [
        "words_per_msg",
        "emoji_per_msg",
        "punct_per_msg",
        "caps_per_word",
        "media_per_msg",
        "exclamation_per_msg",
        "ellipses_per_msg",
        "pct_ends_emoji",
        "pct_ends_punct",
        "pct_has_capitals",
        "mean_response",
        "pct_replies",
    ]
    X = agg[style_cols].fillna(0)
    agg = agg.join(X.add_prefix("style_"))
    return agg, X, style_cols


# ------------------------------------------------------------------
# 6. Dimensionality reduction
# ------------------------------------------------------------------
def reduce_dimensions(X: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca2d = pca.fit_transform(X_scaled)

    # t-SNE (initialised with PCA for stability)
    tsne = TSNE(
        n_components=2,
        perplexity=min(15, len(X) // 3),
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    tsne2d = tsne.fit_transform(X_scaled)

    # UMAP (usually the best visualiser for style clusters)
    umap2d = umap.UMAP(
        n_neighbors=max(2, min(5, len(X) // 4)), min_dist=0.3, random_state=42
    ).fit_transform(X_scaled)

    red = pd.DataFrame(
        {
            "pca_x": pca2d[:, 0],
            "pca_y": pca2d[:, 1],
            "tsne_x": tsne2d[:, 0],
            "tsne_y": tsne2d[:, 1],
            "umap_x": umap2d[:, 0],
            "umap_y": umap2d[:, 1],
        }
    )
    return red


# ------------------------------------------------------------------
# 7. (Optional) HDBSCAN clustering → human-readable style names
# ------------------------------------------------------------------
def cluster_styles(X_scaled: np.ndarray, agg: pd.DataFrame) -> pd.DataFrame:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
    labels = clusterer.fit_predict(X_scaled)
    agg["style_cluster"] = labels

    # Map numbers → friendly names (you can edit)
    name_map = {
        0: "Expressive / Playful",
        1: "Intense / Shouter",
        2: "Reflective / Visual",
        3: "Formal / Structured",
        -1: "Unique / Outlier",
    }
    agg["style_name"] = agg["style_cluster"].map(name_map).fillna("Unique")
    return agg


# ------------------------------------------------------------------
# 8a. Plotly interactive visualisation
# ------------------------------------------------------------------
def plot_umap(agg: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # label = author (or author+year)
    agg["label"] = agg[Columns.AUTHOR.value]
    if Columns.YEAR.value in agg.columns:
        agg["label"] = agg[Columns.AUTHOR.value] + " (" + agg[Columns.YEAR.value].astype(str) + ")"

    fig = px.scatter(
        agg,
        x="umap_x",
        y="umap_y",
        color=Columns.AUTHOR.value,
        size="msg_count",
        hover_data={
            "style_emoji_per_msg": ":.3f",
            "style_caps_per_word": ":.3f",
            "style_exclamation_per_msg": ":.3f",
            "style_ellipses_per_msg": ":.3f",
            "style_words_per_msg": ":.1f",
            "style_name": True,
        },
        title="Linguistic Style Clusters (UMAP)",
        labels={"umap_x": "Style Axis 1", "umap_y": "Style Axis 2"},
        width=950,
        height=650,
    )
    fig.update_traces(marker={"opacity": 0.85, "line": {"width": 1, "color": "black"}})
    html_path = out_dir / "style_umap.html"
    fig.write_html(str(html_path))
    logger.success(f"UMAP plot saved → {html_path}")
    fig.show()


# ------------------------------------------------------------------
# 8b. Plotly interactive visualisation
# ------------------------------------------------------------------
def plot_pca(agg: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    agg["label"] = agg[Columns.AUTHOR.value]
    if Columns.YEAR.value in agg.columns:
        agg["label"] = agg[Columns.AUTHOR.value] + " (" + agg[Columns.YEAR.value].astype(str) + ")"

    fig = px.scatter(
        agg,
        x="pca_x",
        y="pca_y",  # ← Use PCA
        color=Columns.AUTHOR.value,
        size="msg_count",
        hover_data={
            "style_emoji_per_msg": ":.3f",
            "style_caps_per_word": ":.3f",
            "style_exclamation_per_msg": ":.3f",
            "style_ellipses_per_msg": ":.3f",
            "style_words_per_msg": ":.1f",
            "style_name": True,
        },
        title="Linguistic Style Clusters (PCA)",
        labels={"pca_x": "Principal Component 1", "pca_y": "Principal Component 2"},
        width=950,
        height=650,
    )
    fig.update_traces(marker={"opacity": 0.85, "line": {"width": 1, "color": "black"}})
    html_path = out_dir / "style_pca.html"
    fig.write_html(str(html_path))
    logger.success(f"PCA plot saved → {html_path}")
    fig.show()


# ------------------------------------------------------------------
# 8c. Plotly interactive visualisation
# ------------------------------------------------------------------
def plot_tsne(agg: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    agg["label"] = agg[Columns.AUTHOR.value]

    fig = px.scatter(
        agg,
        x="tsne_x",
        y="tsne_y",
        color=Columns.AUTHOR.value,
        size="msg_count",
        hover_data={
            "style_emoji_per_msg": ":.3f",
            "style_caps_per_word": ":.3f",
            "style_exclamation_per_msg": ":.3f",
            "style_ellipses_per_msg": ":.3f",
            "style_words_per_msg": ":.1f",
            "style_name": True,
        },
        title="Linguistic Style Clusters (t-SNE)",
        labels={"tsne_x": "t-SNE Dimension 1", "tsne_y": "t-SNE Dimension 2"},
        width=950,
        height=650,
    )
    fig.update_traces(marker={"opacity": 0.85, "line": {"width": 1, "color": "black"}})
    html_path = out_dir / "style_tsne.html"
    fig.write_html(str(html_path))
    logger.success(f"t-SNE plot saved → {html_path}")
    fig.show()


# ------------------------------------------------------------------
# 9. Full pipeline (one function call)
# ------------------------------------------------------------------
def run_style_analysis(
    by_year: bool = False, do_clustering: bool = True, out_dir: Path = Path("style_output")
):
    df = load_enriched()

    group_cols = choose_group_cols(by_year=by_year)
    agg = aggregate_basic(df, group_cols)

    # ADD PUNCTUATION PATTERNS
    punct = add_punct_patterns(df, group_cols)

    # MERGE INTO agg
    agg = agg.merge(punct, on="author", how="left").fillna(0)

    # Now build features
    agg, X, _style_cols = build_style_features(agg)

    red = reduce_dimensions(X)
    agg = pd.concat([agg, red], axis=1)

    if do_clustering:
        X_scaled = StandardScaler().fit_transform(X)
        agg = cluster_styles(X_scaled, agg)

    # Save summary table
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "style_summary.csv"
    agg.to_csv(summary_path, index=False)
    logger.success(f"Style summary → {summary_path}")

    plot_tsne(agg, out_dir)

    return agg


# ------------------------------------------------------------------
# Example usage (run from the project root)
# ------------------------------------------------------------------
if __name__ == "__main__":
    agg = run_style_analysis(by_year=False, do_clustering=True, out_dir=Path("style_output"))
