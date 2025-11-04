# === Module Docstring ===
"""
Data Preparation Module

Prepares validated data for all 6 visualizations:
1. Categories: Total messages by group/author (Script1)
2. Time: DAC weekly heartbeat (Script2)
3. Distribution: Emoji frequency (Script3)
4. Arc: Author interactions (Script4)
5. Bubble: Words vs punctuation (Script5)
6. Multi-Dimensional: t-SNE + clustering + embeddings (Script6)

**All Pydantic models are defined here** for clean data contracts.
"""

# === Imports ===
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, Dict

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, model_validator, validator

from .constants import Columns, Groups, Script6ConfigKeys

if TYPE_CHECKING:
    from .data_editor import DataEditor


# === Constants for Hard-coded Values ===
DATE_RANGE_START = datetime(2015, 7, 1, tzinfo=timezone.utc)
DATE_RANGE_END = datetime(2025, 7, 31, tzinfo=timezone.utc)
WEEKS_IN_YEAR = range(1, 53)
MAAP_EXPECTED_AUTHORS = 4
TSNE_PERPLEXITY_MAX = 30
HDBSCAN_MIN_CLUSTER_SIZE = 3


# === 1. Category Plot Data Contracts (Script1) ===
class AuthorMessages(BaseModel):
    author: str = Field(..., min_length=1)
    message_count: int = Field(..., ge=0)
    is_avt: bool = False

    class Config:
        extra = "forbid"


class GroupMessages(BaseModel):
    whatsapp_group: Literal["dac", "golfmaten", "maap", "tillies"]
    authors: list[AuthorMessages] = Field(..., min_items=1)
    group_avg: float = Field(..., ge=0.0)

    @property
    def total_messages(self) -> int:
        return sum(a.message_count for a in self.authors)

    class Config:
        extra = "forbid"


class CategoryPlotData(BaseModel):
    groups: list[GroupMessages]
    total_messages: int
    date_range: tuple[datetime, datetime]

    @property
    def group_order(self) -> list[str]:
        return [g.whatsapp_group for g in self.groups]

    @model_validator(mode="after")
    def check_total_matches_sum(self):
        expected = sum(g.total_messages for g in self.groups)
        if self.total_messages != expected:
            raise ValueError("total_messages must equal sum of all GroupMessages.total_messages")
        return self

    class Config:
        extra = "forbid"


# === 2. Time Plot Data Contract (Script2) ===
class TimePlotData(BaseModel):
    weekly_avg: Dict[int, float]
    global_avg: float
    date_range: tuple[datetime, datetime]

    @validator("weekly_avg")
    def validate_week_keys(cls, v):
        invalid = [k for k in v.keys() if k not in WEEKS_IN_YEAR]
        if invalid:
            raise ValueError(f"Weekly keys must be in 1–52, got invalid: {invalid}")
        return v

    class Config:
        arbitrary_types_allowed = False
        extra = "forbid"


# === 3. Distribution Plot Data Contract (Script3) ===
class DistributionPlotData(BaseModel):
    """Validated container for the emoji-frequency DataFrame."""
    emoji_counts_df: pd.DataFrame = Field(
        ..., description="Columns: ['emoji', 'count_once', 'percent_once']"
    )

    @model_validator(mode="after")
    def validate_columns(self):
        required = {"emoji", "count_once", "percent_once"}
        if not required.issubset(self.emoji_counts_df.columns):
            missing = required - set(self.emoji_counts_df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        return self

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


# === 4. Arc Plot Data Contract (Script4) ===
class ArcPlotData(BaseModel):
    """Validated container for the participation table used by the arc diagram."""
    participation_df: pd.DataFrame = Field(
        ..., description="Columns: ['type', 'author', 'total_messages', +author_names]"
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


# === 5. Bubble Plot Data Contract (Script5) ===
class BubblePlotData(BaseModel):
    """Validated container for the feature table used by the bubble plot."""
    feature_df: pd.DataFrame = Field(
        ...,
        description="Columns: ['whatsapp_group', 'author', 'avg_words', 'avg_punct', 'message_count']"
    )

    @model_validator(mode="after")
    def validate_columns(self):
        required = {
            Columns.WHATSAPP_GROUP.value,
            Columns.AUTHOR.value,
            Columns.AVG_WORDS.value,
            Columns.AVG_PUNCT.value,
            Columns.MESSAGE_COUNT.value,
        }
        if not required.issubset(self.feature_df.columns):
            missing = required - set(self.feature_df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        return self

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


# === 6. Multi Dimensional Plot (Script6) ===
class EmbeddingModel(int):
    STYLE = 1
    MINILM = 2
    MPNET = 3


MODEL_NAME_MAP = {
    EmbeddingModel.STYLE: "AnnaWegmann/Style-Embedding",
    EmbeddingModel.MINILM: "sentence-transformers/all-MiniLM-L6-v2",
    EmbeddingModel.MPNET: "sentence-transformers/all-mpnet-base-v2",
}

class MultiDimPlotData(BaseModel):
    """Validated container for t-SNE + clustering + embeddings."""
    agg_df: pd.DataFrame = Field(..., description="Aggregated features with t-SNE coords and clusters")
    style_cols: list[str] = Field(..., description="Hand-crafted style feature column names")
    model_used: str = Field(..., description="Embedding model identifier")
    by_year: bool = True

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


class MultiDimPlotSettings(BaseModel):
    """Configuration for multi-dimensional t-SNE visualization (Script6)."""
    by_group: bool = True
    draw_ellipses: bool = False
    use_embeddings: bool = True
    hybrid_features: bool = True
    embedding_model: int = 3  # 1=Style, 2=MiniLM, 3=MPNet


# === Base Handler ===
class BaseHandler:
    def _log_debug(self, message: str) -> None:
        logger.debug(message)

    def _handle_empty_df(self, df: pd.DataFrame, context: str) -> pd.DataFrame:
        if df.empty:
            logger.error(f"Empty DataFrame in {context}.")
            return pd.DataFrame()
        return df


# === Data Preparation Class ===
class DataPreparation(BaseHandler):
    def __init__(self, data_editor: DataEditor | None = None) -> None:
        self.data_editor = data_editor


    # === 1. Categories (Script1) ===
    def build_visual_categories(self, df: pd.DataFrame) -> CategoryPlotData | None:
        """
        Build category plot data: messages per author per group, with AvT highlighted.

        Args:
            df: Input DataFrame with required columns.

        Returns:
            CategoryPlotData with grouped author stats or None on failure.

        Raises:
            Exception: If processing fails (logged).
        """
        df = self._handle_empty_df(df, "build_visual_categories")
        if df.empty:
            return None

        try:
            mask = (df[Columns.TIMESTAMP] >= DATE_RANGE_START) & (df[Columns.TIMESTAMP] <= DATE_RANGE_END)
            df_filtered = df[mask].copy()
            if df_filtered.empty:
                logger.warning(f"No messages in date range {DATE_RANGE_START.date()} to {DATE_RANGE_END.date()}")
                return None

            counts = (
                df_filtered.groupby([Columns.WHATSAPP_GROUP, Columns.AUTHOR], as_index=False)
                .size()
                .rename(columns={"size": Columns.MESSAGE_COUNT})
            )
            if counts.empty:
                logger.error("No message counts after grouping")
                return None

            group_order = sorted(counts[Columns.WHATSAPP_GROUP].unique())
            counts[Columns.WHATSAPP_GROUP] = pd.Categorical(
                counts[Columns.WHATSAPP_GROUP],
                categories=group_order,
                ordered=True,
            )
            counts = counts.sort_values(Columns.WHATSAPP_GROUP).reset_index(drop=True)
            counts["is_avt"] = counts[Columns.AUTHOR] == Groups.AVT

            groups_data = []
            total_messages = 0

            for group in group_order:
                grp = counts[counts[Columns.WHATSAPP_GROUP] == group]
                non_avt = grp[~grp["is_avt"]].sort_values(Columns.MESSAGE_COUNT, ascending=False)
                avt_row = grp[grp["is_avt"]]
                group_avg = non_avt[Columns.MESSAGE_COUNT].mean() if not non_avt.empty else 0.0
                authors = []

                for _, row in non_avt.iterrows():
                    msg_count = int(row[Columns.MESSAGE_COUNT])
                    authors.append(
                        AuthorMessages(
                            author=str(row[Columns.AUTHOR]),
                            message_count=msg_count,
                            is_avt=False,
                        )
                    )
                    total_messages += msg_count

                if not avt_row.empty:
                    row = avt_row.iloc[0]
                    msg_count = int(row[Columns.MESSAGE_COUNT])
                    authors.append(
                        AuthorMessages(
                            author=str(row[Columns.AUTHOR]),
                            message_count=msg_count,
                            is_avt=True,
                        )
                    )
                    total_messages += msg_count

                groups_data.append(
                    GroupMessages(
                        whatsapp_group=group,
                        authors=authors,
                        group_avg=float(group_avg),
                    )
                )

            result = CategoryPlotData(
                groups=groups_data,
                total_messages=total_messages,
                date_range=(DATE_RANGE_START, DATE_RANGE_END),
            )

            logger.success(
                f"CategoryPlotData built: {len(result.groups)} groups, "
                f"{result.total_messages:,} messages"
            )
            return result

        except Exception as e:
            logger.exception(f"build_visual_categories failed: {e}")
            return None


    # === 2. Time (Script2) ===
    def build_visual_time(self, df: pd.DataFrame) -> TimePlotData | None:
        """
        Build weekly message average for DAC group.

        Args:
            df: Input DataFrame.

        Returns:
            TimePlotData with weekly averages or None.

        Raises:
            Exception: If processing fails (logged).
        """
        df = self._handle_empty_df(df, "build_visual_time")
        if df.empty:
            return None

        try:
            df_dac = df[df[Columns.WHATSAPP_GROUP] == Groups.DAC].copy()
            if df_dac.empty:
                logger.warning(f"No messages for group '{Groups.DAC.value}'. Skipping.")
                return None

            week_col = Columns.WEEK.value
            if week_col not in df_dac.columns:
                logger.error(f"Missing required column '{week_col}' in DAC DataFrame.")
                return None

            # Weekly message count per author
            weekly_counts = df_dac.groupby(week_col).size()

            # Number of unique authors per week
            authors_per_week = df_dac.groupby(week_col)[Columns.AUTHOR.value].nunique()

            # Average messages per author per week (rounded to 1 decimal)
            weekly_avg = (weekly_counts / authors_per_week).round(1)

            # Fill missing weeks (1–52) with 0.0
            weekly_avg = weekly_avg.reindex(WEEKS_IN_YEAR, fill_value=0.0)

            data = TimePlotData(
                weekly_avg=weekly_avg.to_dict(),
                global_avg=float(weekly_avg.mean()),
                date_range=(
                    df_dac[Columns.TIMESTAMP].min(),
                    df_dac[Columns.TIMESTAMP].max(),
                ),
            )

            logger.success(
                f"TimePlotData built - {len(weekly_avg)} weeks, "
                f"global avg {data.global_avg:.1f}"
            )
            return data

        except Exception as e:
            logger.exception(f"build_visual_time failed: {e}")
            return None


    # === 3. Distribution (Script3) ===
    def build_visual_distribution(self, df: pd.DataFrame) -> DistributionPlotData | None:
        """
        Build emoji frequency table.

        Args:
            df: DataFrame with LIST_OF_ALL_EMOJIS column.

        Returns:
            DistributionPlotData or None.

        Raises:
            Exception: If parsing or counting fails.
        """
        df = self._handle_empty_df(df, "build_visual_distribution")
        if df.empty:
            return None

        try:
            import ast
            import emoji

            def parse_emoji_list(cell):
                if pd.isna(cell) or cell in {'[]', '', ' ', None}:
                    return []
                try:
                    return ast.literal_eval(cell)
                except (ValueError, SyntaxError):
                    return [e.strip() for e in str(cell).split() if e.strip() in emoji.EMOJI_DATA]

            emoji_lists = df[Columns.LIST_OF_ALL_EMOJIS.value].apply(parse_emoji_list)
            all_emojis = pd.Series([e for sublist in emoji_lists for e in sublist])

            if all_emojis.empty:
                logger.warning("No emojis found after parsing.")
                return None

            emoji_counts = Counter(all_emojis)
            emoji_counts_df = pd.DataFrame(
                {
                    "emoji": list(emoji_counts.keys()),
                    "count_once": list(emoji_counts.values()),
                }
            )
            emoji_counts_df["percent_once"] = (
                emoji_counts_df["count_once"] / len(df) * 100
            )
            emoji_counts_df = emoji_counts_df.sort_values(
                by="count_once", ascending=False
            ).reset_index(drop=True)

            result = DistributionPlotData(emoji_counts_df=emoji_counts_df)
            logger.success(
                f"DistributionPlotData built – {len(emoji_counts_df)} unique emojis."
            )
            return result

        except Exception as e:
            logger.exception(f"build_visual_distribution failed: {e}")
            return None


    # === 4. Arc (Script4) ===
    def build_visual_relationships_arc(self, df_group: pd.DataFrame) -> ArcPlotData | None:
        """
        Build the participation table for the MAAP arc diagram.

        Uses ``x_number_of_unique_participants_that_day`` to identify 2- or 3-person days.
        Authors are derived from the data (must be exactly 4).

        Args:
            df_group: DataFrame filtered to MAAP group.

        Returns:
            ArcPlotData or None.

        Raises:
            Exception: If processing fails.
        """
        df = self._handle_empty_df(df_group, "build_visual_relationships_arc")
        if df.empty:
            return None

        try:
            authors = sorted(df[Columns.AUTHOR].unique())
            if len(authors) != MAAP_EXPECTED_AUTHORS:
                logger.error(f"MAAP group must have exactly {MAAP_EXPECTED_AUTHORS} authors, found {len(authors)}")
                return None

            df = df.copy()
            df["date"] = df[Columns.TIMESTAMP].dt.date

            rows = []

            for day, day_df in df.groupby("date"):
                n_part = day_df[Columns.X_NUMBER_OF_UNIQUE_PARTICIPANTS_THAT_DAY].iloc[0]
                if n_part not in (2, 3):
                    continue

                total_messages = len(day_df)
                author_counts = day_df[Columns.AUTHOR].value_counts()
                pct = {a: (author_counts.get(a, 0) / total_messages) * 100 for a in authors}

                if n_part == 2:
                    active = sorted([a for a in authors if pct[a] > 0])
                    author_label = " & ".join(active)
                    row = {
                        "type": "Pairs",
                        "author": author_label,
                        "total_messages": total_messages,
                    }
                    for a in authors:
                        row[a] = f"{pct[a]:.0f}%" if a in active else 0

                else:  # n_part == 3
                    missing = [a for a in authors if pct[a] == 0][0]
                    author_label = f"Missing: {missing}"
                    row = {
                        "type": "Non-participant",
                        "author": author_label,
                        "total_messages": total_messages,
                    }
                    for a in authors:
                        row[a] = f"{pct[a]:.0f}%" if pct[a] > 0 else 0

                rows.append(row)

            if not rows:
                logger.error("No qualifying days found for arc diagram.")
                return None

            participation_df = pd.DataFrame(rows)
            col_order = ["type", "author", "total_messages"] + authors
            participation_df = participation_df[col_order]

            logger.success(f"Arc table built: {len(participation_df)} rows")
            return ArcPlotData(participation_df=participation_df)

        except Exception as e:
            logger.exception(f"build_visual_relationships_arc failed: {e}")
            return None


    # === 5. Bubble (Script5) ===
    def build_visual_relationships_bubble(self, df_groups: pd.DataFrame) -> BubblePlotData | None:
        """
        Build bubble plot features: avg words, punctuation, message count per author-group.

        Args:
            df_groups: DataFrame with group, author, word/punct counts.

        Returns:
            BubblePlotData or None.

        Raises:
            Exception: If aggregation fails.
        """
        df = self._handle_empty_df(df_groups, "build_visual_relationships_bubble")
        if df.empty:
            return None

        try:
            required_cols = [
                Columns.WHATSAPP_GROUP.value,
                Columns.AUTHOR.value,
                Columns.NUMBER_OF_WORDS.value,
                Columns.NUMBER_OF_PUNCTUATIONS.value,
            ]
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Bubble plot missing required input columns: {required_cols}")
                return None

            agg_df = (
                df.groupby([Columns.WHATSAPP_GROUP.value, Columns.AUTHOR.value])
                .agg(
                    **{
                        Columns.AVG_WORDS.value: (Columns.NUMBER_OF_WORDS.value, "mean"),
                        Columns.AVG_PUNCT.value: (Columns.NUMBER_OF_PUNCTUATIONS.value, "mean"),
                        Columns.MESSAGE_COUNT.value: (Columns.AUTHOR.value, "size"),
                    }
                )
                .reset_index()
            )

            if agg_df.empty:
                logger.error("No author-group combinations after aggregation.")
                return None

            logger.success(
                f"Bubble feature table built – {len(agg_df)} author-group rows"
            )
            return BubblePlotData(feature_df=agg_df)

        except Exception as e:
            logger.exception(f"build_visual_relationships_bubble failed: {e}")
            return None


    # === 6. Multi-Dimensional Style (Script6) ===
    def build_visual_multi_dimensions(
        self,
        df: pd.DataFrame,
        settings: dict,
    ) -> MultiDimPlotData | None:
        """
        Build aggregated style features, embeddings, t-SNE, and clustering.

        Args:
            df: Enriched DataFrame
            settings: Dict with by_group, use_embeddings, hybrid_features, embedding_model

        Returns:
            MultiDimPlotData or None.

        Raises:
            Exception: If any step fails.
        """
        df = self._handle_empty_df(df, "build_visual_multi_dimensions")
        if df.empty:
            return None

        try:
            # Rename columns to standard names
            df = df.copy()
            rename_map = {}
            for col in df.columns:
                lower = col.lower().strip()
                if lower == "author":
                    rename_map[col] = Columns.AUTHOR.value
                elif lower == "year":
                    rename_map[col] = Columns.YEAR.value
                elif lower == "whatsapp_group":
                    rename_map[col] = Columns.WHATSAPP_GROUP.value
            if rename_map:
                df = df.rename(columns=rename_map)

            if "message_cleaned" not in df.columns:
                df["message_cleaned"] = ""

            # Isolate AvT into temp group
            df[Columns.WHATSAPP_GROUP_TEMP] = df[Columns.WHATSAPP_GROUP.value]
            df.loc[df[Columns.AUTHOR.value] == Groups.AVT, Columns.WHATSAPP_GROUP_TEMP] = Groups.AVT

            # Aggregation keys
            group_cols = [Columns.AUTHOR.value, Columns.YEAR.value]
            full_group_cols = [*group_cols, Columns.WHATSAPP_GROUP_TEMP]

            agg_dict = {
                "msg_count": ("message_cleaned", "count"),
                "words_total": (Columns.NUMBER_OF_WORDS.value, "sum"),
                "emojis_total": (Columns.NUMBER_OF_EMOJIS.value, "sum"),
                "punct_total": (Columns.NUMBER_OF_PUNCTUATIONS.value, "sum"),
                "caps_total": (Columns.NUMBER_OF_CAPITALS.value, "sum"),
                "pics_total": (Columns.NUMBER_OF_PICTURES_VIDEOS.value, "sum"),
                "links_total": (Columns.HAS_LINK.value, "sum"),
                "attachments_total": (Columns.HAS_ATTACHMENT.value, "sum"),
                "mean_response": (Columns.RESPONSE_TIME.value, "mean"),
                "std_response": (Columns.RESPONSE_TIME.value, "std"),
                "pct_ends_emoji": (Columns.ENDS_WITH_EMOJI.value, "mean"),
                "pct_ends_punct": (Columns.ENDS_WITH_PUNCTUATION.value, "mean"),
                "pct_has_capitals": (Columns.HAS_CAPITALS.value, "mean"),
                "avg_word_len": (Columns.AVG_WORD_LENGTH.value, "mean"),
                "mean_chat_len": (Columns.LENGTH_CHAT.value, "mean"),
                "mean_caps_ratio": (Columns.CAPITALIZED_WORDS_RATIO.value, "mean"),
                "pct_starts_capital": (Columns.STARTS_WITH_CAPITAL.value, "mean"),
                "mean_pct_emojis": (Columns.PCT_EMOJIS.value, "mean"),
                "pct_starts_emoji": (Columns.STARTS_WITH_EMOJI.value, "mean"),
                "mean_pct_punct": (Columns.PCT_PUNCTUATIONS.value, "mean"),
                "pct_has_question": (Columns.HAS_QUESTION_MARK.value, "mean"),
                "pct_ends_question": (Columns.ENDS_WITH_QUESTION_MARK.value, "mean"),
                "numbers_total": (Columns.NUMBER_OF_NUMBERS.value, "sum"),
                "pct_was_deleted": (Columns.WAS_DELETED.value, "mean"),
            }

            agg = df.groupby(full_group_cols).agg(**agg_dict).reset_index()

            # Compute pct_replies
            prev = df[Columns.PREVIOUS_AUTHOR.value]
            curr = df[Columns.AUTHOR.value]
            reply_mask = prev.notna() & (prev != curr)
            group_key = df[full_group_cols].apply(tuple, axis=1)
            reply_pct = reply_mask.groupby(group_key).mean()
            reply_df = reply_pct.reset_index(name="pct_replies")
            reply_df[full_group_cols] = pd.DataFrame(reply_df["index"].tolist(), columns=full_group_cols)
            reply_df = reply_df.drop(columns="index")
            agg = agg.merge(reply_df, on=full_group_cols, how="left").fillna(0)

            # Style features
            style_cols = [
                "msg_count", "words_total", "emojis_total", "punct_total", "caps_total",
                "pics_total", "links_total", "attachments_total", "mean_response", "std_response",
                "pct_ends_emoji", "pct_ends_punct", "pct_has_capitals", "avg_word_len",
                "mean_chat_len", "mean_caps_ratio", "pct_starts_capital", "mean_pct_emojis",
                "pct_starts_emoji", "mean_pct_punct", "pct_has_question", "pct_ends_question",
                "numbers_total", "pct_was_deleted", "pct_replies"
            ]
            X = agg[style_cols].copy()

            # Embedding settings
            use_emb = settings.get(Script6ConfigKeys.USE_EMBEDDINGS, False)
            hybrid = settings.get(Script6ConfigKeys.HYBRID_FEATURES, True)
            model_id = settings.get(Script6ConfigKeys.EMBEDDING_MODEL, EmbeddingModel.MPNET)

            model_name = MODEL_NAME_MAP.get(model_id, MODEL_NAME_MAP[EmbeddingModel.MPNET])

            if use_emb:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {model_name}")
                model = SentenceTransformer(model_name)
                texts = df["message_cleaned"].fillna("").tolist()
                embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
                emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
                emb_df[full_group_cols] = df[full_group_cols].reset_index(drop=True)
                emb_agg = emb_df.groupby(full_group_cols).mean().reset_index()
                agg = agg.merge(emb_agg, on=full_group_cols, how="left")
                emb_cols = [c for c in agg.columns if c.startswith("emb_")]
                if hybrid:
                    X = pd.concat([X, agg[emb_cols].fillna(0)], axis=1)
                else:
                    X = agg[emb_cols].fillna(0)
                    style_cols = emb_cols

            # Dimension reduction
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            X_scaled = StandardScaler().fit_transform(X)
            
            n_features = X.shape[1]
            n_components = min(50, n_features)
            logger.info(f"PCA using {n_components} components (max available: {n_features})")
            
            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            from sklearn.manifold import TSNE
            perplexity = min(TSNE_PERPLEXITY_MAX, len(X) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_coords = tsne.fit_transform(X_pca)
            agg["tsne_x"] = tsne_coords[:, 0]
            agg["tsne_y"] = tsne_coords[:, 1]

            # Clustering
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, metric='euclidean')
            agg["cluster"] = clusterer.fit_predict(X_scaled)

            result = MultiDimPlotData(
                agg_df=agg,
                style_cols=style_cols,
                model_used=model_name,
                by_year=True,
            )
            logger.success(f"MultiDimPlotData built: {len(agg)} rows, {len(style_cols)} features")
            return result

        except Exception as e:
            logger.exception(f"build_visual_multi_dimensions failed: {e}")
            return None


# === Public API ===
__all__ = [
    "CategoryPlotData",
    "TimePlotData",
    "DistributionPlotData",
    "ArcPlotData",
    "BubblePlotData",
    "MultiDimPlotData",
    "MultiDimPlotSettings",
    "DataPreparation",
]


# === CODING STANDARD ===
# - `# === Module Docstring ===` before """
# - Google-style docstrings
# - `# === Section Name ===` for all blocks
# - Inline: `# One space, sentence case`
# - Tags: `# TODO:`, `# NOTE:`, `# NEW: (YYYY-MM-DD)`, `# FIXME:`
# - Type hints in function signatures
# - Examples: with >>>
# - No long ----- lines
# - No mixed styles
# - Add markers #NEW at the end of the module

# NEW: Migrated @root_validator → @model_validator (Pydantic v2) (2025-11-03)
# NEW: All previous improvements preserved: constants, column checks, strict models