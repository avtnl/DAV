# === Module Docstring ===
"""
Data Preparation Module

Prepares validated data for all 5 visualizations:
1. Categories: Total messages by group/author (Script1)
2. Time: DAC weekly heartbeat (Script2)
3. Distribution: Emoji frequency (Script3)
4. Relationship: Words vs punctuation (Script4)
5. Multi-Dimensional: t-SNE + clustering + embeddings (Script5)

**All Pydantic models are defined here** for clean data contracts.
"""

# === Imports ===
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, Dict, Optional

import ast
import pandas as pd
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, model_validator, validator
from emoji import demojize

from .constants import Columns, Groups, Script5ConfigKeys

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
    active_years: int = Field(0, ge=0)

    class Config:
        extra = "forbid"


class GroupMessages(BaseModel):
    whatsapp_group: str
    authors: list[AuthorMessages]
    group_avg: float = Field(..., description="Expected TOTAL messages for average non-AvT member")
    avt_total: int = Field(0, description="AvT's actual total messages")

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
    seasonality: Optional["SeasonalityEvidence"] = None  # Forward reference

    @validator("weekly_avg")
    def validate_week_keys(cls, v):
        invalid = [k for k in v.keys() if k not in WEEKS_IN_YEAR]
        if invalid:
            raise ValueError(f"Weekly keys must be in 1–52, got invalid: {invalid}")
        return v

    class Config:
        arbitrary_types_allowed = False
        extra = "forbid"


class SeasonalityEvidence(BaseModel):
    """All statistical proof that the weekly series is seasonal + low-noise."""
    acf: list[float] = Field(..., description="ACF values (lags 0 to max)")
    decomposition: Dict[str, list[float]] = Field(
        ..., description="trend, seasonal, resid as lists (length = full series)"
    )
    fourier: Dict[str, list[float]] = Field(
        ..., description="freqs, amps, phases as lists for the k strongest components"
    )
    filtered: Dict[str, list[float]] = Field(
        ..., description="savitzky_golay, butterworth_lowpass as lists (length = full series)"
    )
    residual_std: float = Field(..., description="Std-dev of decomposition residuals (nanstd)")
    dominant_period_weeks: int = Field(
        ..., description="Period of the strongest non-zero Fourier component (rounded)"
    )
    raw_series: list[float] = Field(..., description="Full chronological weekly counts")

    class Config:
        extra = "forbid"


# === 3. Distribution Plot Data Contract (Script3) ===
class DistributionPlotData(BaseModel):
    """Validated container for the emoji-frequency DataFrame."""
    emoji_counts_df: pd.DataFrame = Field(
        ..., description="Columns: ['emoji', 'count_once', 'percent_once', 'unicode_code', 'unicode_name']"
    )

    @model_validator(mode="after")
    def validate_columns(self):
        required = {"emoji", "count_once", "percent_once", "unicode_name", "unicode_code"}
        missing = required - set(self.emoji_counts_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return self

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


# === Power-Law Analysis Models ===
class PowerLawFitResult(BaseModel):
    """Results from power-law fit using powerlaw package."""
    alpha: float = Field(..., description="Power-law exponent α")
    xmin: int = Field(..., description="Lower bound for power-law behavior")
    D: float = Field(..., description="K-S statistic (distance)")
    loglikelihood: float = Field(..., description="Log-likelihood of the power-law model")
    n_tail: int = Field(..., description="Number of observations in tail (≥ xmin)")

    class Config:
        extra = "forbid"


class ModelComparisonResult(BaseModel):
    """Likelihood ratio test p-values vs alternative models."""
    vs_exponential: float = Field(..., description="p-value vs exponential")
    vs_lognormal: float = Field(..., description="p-value vs lognormal")
    R: float = Field(..., description="Log-likelihood ratio (positive = power-law better)")

    class Config:
        extra = "forbid"


class PowerLawAnalysisResult(BaseModel):
    """Full power-law analysis output."""
    fit: PowerLawFitResult
    comparison: ModelComparisonResult
    n_observations: int
    total_messages_with_emoji: int
    top_10_emojis: list[str] = Field(default_factory=list)

    @property
    def is_power_law(self) -> bool:
        """True if power-law beats both alternatives (p < 0.05)."""
        return (
            self.comparison.vs_exponential < 0.05
            and self.comparison.vs_lognormal < 0.05
        )

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True


# === 4. Relationships Plot Data Contract (Script4) ===
class RelationshipsPlotData(BaseModel):
    """Validated container for the feature table used by the relationships plot."""
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


# === 5. Multi Dimensional Plot (Script5) ===
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
    agg_df: pd.DataFrame = Field(..., description="Aggregated features with t-SNE/PCA coords and clusters")
    style_cols: list[str] = Field(..., description="Hand-crafted style feature column names")
    model_used: str = Field(..., description="Embedding model identifier")
    by_year: bool = True
    plot_type: str = Field(..., description="pca | tsne | both")

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


class MultiDimPlotSettings(BaseModel):
    """Configuration for multi-dimensional t-SNE visualization (Script5)."""
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
        Build category plot data: messages per author per group, with fair AvT comparison.

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
            # Convert TIMESTAMP to tz-aware UTC to avoid tz-naive comparison error
            ts = pd.to_datetime(df[Columns.TIMESTAMP], utc=True)
            mask = (ts >= DATE_RANGE_START) & (ts <= DATE_RANGE_END)
            df_filtered = df[mask].copy()
            if df_filtered.empty:
                logger.warning(
                    f"No messages in date range {DATE_RANGE_START.date()} to {DATE_RANGE_END.date()}"
                )
                return None

            # Count messages per author per group
            counts = (
                df_filtered.groupby([Columns.WHATSAPP_GROUP, Columns.AUTHOR], as_index=False)
                .size()
                .rename(columns={"size": Columns.MESSAGE_COUNT})
            )
            if counts.empty:
                logger.error("No message counts after grouping")
                return None

            # Enforce consistent group order
            group_order = sorted(counts[Columns.WHATSAPP_GROUP].unique())
            counts[Columns.WHATSAPP_GROUP] = pd.Categorical(
                counts[Columns.WHATSAPP_GROUP], categories=group_order, ordered=True
            )
            counts = counts.sort_values(Columns.WHATSAPP_GROUP).reset_index(drop=True)
            counts["is_avt"] = counts[Columns.AUTHOR] == Groups.AVT

            # Merge ACTIVE_YEARS for tooltips (optional)
            if Columns.ACTIVE_YEARS in df.columns:
                active_years_map = df[[Columns.AUTHOR, Columns.ACTIVE_YEARS]].drop_duplicates()
                counts = counts.merge(active_years_map, on=Columns.AUTHOR, how="left")
                counts[Columns.ACTIVE_YEARS] = counts[Columns.ACTIVE_YEARS].fillna(0).astype(int)
            else:
                counts[Columns.ACTIVE_YEARS] = 0

            # Build Pydantic containers
            groups_data: list[GroupMessages] = []
            total_messages = 0

            for group in group_order:
                grp = counts[counts[Columns.WHATSAPP_GROUP] == group]
                non_avt = grp[~grp["is_avt"]].sort_values(Columns.MESSAGE_COUNT, ascending=False)
                avt_row = grp[grp["is_avt"]]

                # Compute fair group average: mean of non-AvT total messages
                group_avg = (
                    float(non_avt[Columns.MESSAGE_COUNT].mean())
                    if not non_avt.empty
                    else 0.0
                )

                # Extract AvT total
                avt_total = (
                    int(avt_row.iloc[0][Columns.MESSAGE_COUNT])
                    if not avt_row.empty
                    else 0
                )

                # Build author list with optional active years
                authors: list[AuthorMessages] = []
                for _, row in non_avt.iterrows():
                    msg_cnt = int(row[Columns.MESSAGE_COUNT])
                    authors.append(
                        AuthorMessages(
                            author=str(row[Columns.AUTHOR]),
                            message_count=msg_cnt,
                            is_avt=False,
                            active_years=int(row.get(Columns.ACTIVE_YEARS, 0)),
                        )
                    )
                    total_messages += msg_cnt

                if not avt_row.empty:
                    row = avt_row.iloc[0]
                    msg_cnt = int(row[Columns.MESSAGE_COUNT])
                    authors.append(
                        AuthorMessages(
                            author=str(row[Columns.AUTHOR]),
                            message_count=msg_cnt,
                            is_avt=True,
                            active_years=int(row.get(Columns.ACTIVE_YEARS, 0)),
                        )
                    )
                    total_messages += msg_cnt

                # Pass fair average and AvT total to plot
                groups_data.append(
                    GroupMessages(
                        whatsapp_group=group,
                        authors=authors,
                        group_avg=group_avg,
                        avt_total=avt_total,
                    )
                )

            # Final validated container
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


    # === Build Functions (Script2) ===
    def build_visual_time(
        self,
        df: pd.DataFrame,
        compute_seasonality: bool = True,
    ) -> TimePlotData:
        """Build TimePlotData for DAC weekly averages, optionally with seasonality evidence."""
        df_dac = df[df[Columns.WHATSAPP_GROUP.value] == Groups.DAC.value]
        if df_dac.empty:
            raise ValueError("No DAC data")

        df_dac = df_dac.copy()
        df_dac["week"] = df_dac[Columns.TIMESTAMP.value].dt.isocalendar().week
        weekly_counts = df_dac.groupby("week").size()
        weekly_avg = {week: weekly_counts.get(week, 0.0) for week in WEEKS_IN_YEAR}
        global_avg = sum(weekly_avg.values()) / len(weekly_avg)

        date_range = (
            df_dac[Columns.TIMESTAMP.value].min(),
            df_dac[Columns.TIMESTAMP.value].max(),
        )

        data = TimePlotData(
            weekly_avg=weekly_avg,
            global_avg=global_avg,
            date_range=date_range,
        )

        if compute_seasonality:
            data.seasonality = self._compute_seasonality_evidence(df_dac)

        return data

    def _weekly_series(self, df_dac: pd.DataFrame) -> pd.Series:
        """Return a complete weekly count series (weeks 1–52 for every year)."""
        df = df_dac.copy()
        df["year"] = df[Columns.TIMESTAMP.value].dt.year
        df["week"] = df[Columns.TIMESTAMP.value].dt.isocalendar().week

        years = df["year"].unique()
        idx = pd.MultiIndex.from_product([years, range(1, 53)], names=["year", "week"])
        full = pd.DataFrame(index=idx).reset_index()

        cnt = df.groupby(["year", "week"]).size().reset_index(name="count")
        cnt = full.merge(cnt, on=["year", "week"], how="left").fillna({"count": 0})
        return pd.Series(cnt["count"].values, name="weekly_count")

    def _compute_seasonality_evidence(
        self,
        df_dac: pd.DataFrame,
        k_fourier: int = 3,
        savgol_window: int = 7,
        butter_order: int = 3,
        butter_cutoff_year: float = 1.0,
    ) -> SeasonalityEvidence:
        from statsmodels.tsa.stattools import acf
        from statsmodels.tsa.seasonal import seasonal_decompose
        from numpy.fft import fft, fftfreq
        from scipy.signal import butter, filtfilt, savgol_filter

        series = self._weekly_series(df_dac)
        y = series.values.astype(float)
        n = len(y)

        acf_vals = acf(y, nlags=min(52, n // 2), fft=True)
        acf_list = acf_vals.tolist()

        start = pd.to_datetime(f"{df_dac[Columns.TIMESTAMP.value].dt.year.min()}-01-01")
        idx = pd.date_range(start, periods=n, freq="W-MON")
        ts = pd.Series(y +0.1, index=idx)
        decomp = seasonal_decompose(ts, model="multiplicative", period=52)
        decomp_dict = {
            "trend": decomp.trend.values.tolist(),
            "seasonal": decomp.seasonal.values.tolist(),
            "resid": decomp.resid.values.tolist(),
        }

        # === Print residual stats ===
        import numpy as np

        # 1. Raw additive model
        ts_raw = pd.Series(y, index=idx)
        decomp_raw = seasonal_decompose(ts_raw, model="additive", period=52)
        resid_std_raw = np.nanstd(decomp_raw.resid)

        # 2. Log + additive model
        y_log = np.log1p(y)
        ts_log = pd.Series(y_log, index=idx)
        decomp_log = seasonal_decompose(ts_log, model="additive", period=52)
        resid_std_log = np.nanstd(decomp_log.resid)
        resid_std_back = np.expm1(resid_std_log)  # approximate back-transform

        # Print results
        print(f"{'Model':<25} {'Residual σ':<15} {'Back to messages'}")
        print(f"{'-'*50}")
        print(f"{'Raw + additive':<25} {resid_std_raw:<15.2f} {'—'}")
        print(f"{'Log + additive':<25} {resid_std_log:<15.3f} {resid_std_back:>8.1f}")

        yf = fft(y)
        freqs = fftfreq(n, d=1.0)[: n // 2]
        amps = 2.0 / n * np.abs(yf[: n // 2])
        phases = np.angle(yf[: n // 2]) + np.pi / 2

        top_idx = np.argsort(amps)[-k_fourier:]
        fourier_dict = {
            "freqs": freqs[top_idx].tolist(),
            "amps": amps[top_idx].tolist(),
            "phases": phases[top_idx].tolist(),
        }
        non_zero_idx = top_idx[freqs[top_idx] > 0]
        dominant_period = int(round(1.0 / freqs[non_zero_idx[-1]])) if non_zero_idx.size else 0

        win = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
        win = min(win, n)
        sg = savgol_filter(y, window_length=win, polyorder=3)

        nyq = 0.5
        cutoff = butter_cutoff_year / 52.0
        b, a = butter(butter_order, cutoff / nyq, btype="low")
        bw = filtfilt(b, a, y)

        filtered_dict = {"savitzky_golay": sg.tolist(), "butterworth": bw.tolist()}
        resid_std = float(np.nanstd(decomp.resid))

        return SeasonalityEvidence(
            acf=acf_list,
            decomposition=decomp_dict,
            fourier=fourier_dict,
            filtered=filtered_dict,
            residual_std=resid_std,
            dominant_period_weeks=dominant_period,
            raw_series=y.tolist(),
        )

    # === 3. Distribution (Script3) ===

    # === Power-Law Analysis Function ===
    def analyze_emoji_distribution_power_law(
        self,
        df: pd.DataFrame,
        min_count_for_tail: int = 5,
        max_xmin: int = 50,
    ) -> PowerLawAnalysisResult | None:
        """
        Perform rigorous power-law (Zipf) analysis on emoji frequencies.

        Uses `powerlaw` package (Clauset et al., 2009) with:
        - Automatic xmin selection
        - K-S distance
        - Log-likelihood (summed over tail)
        - Likelihood-ratio tests vs exponential & lognormal

        Args:
            df: Input DataFrame (same as build_visual_distribution)
            min_count_for_tail: Minimum count to consider for tail (default 5)
            max_xmin: Upper bound for xmin search

        Returns:
            PowerLawAnalysisResult or None on failure
        """
        df = self._handle_empty_df(df, "analyze_emoji_distribution_power_law")
        if df.empty:
            return None

        try:
            # Reuse existing emoji parsing
            emojis_list = self.data_editor.parse_emojis(df)
            if not emojis_list.any():
                logger.warning("No emojis found for power-law analysis")
                return None

            all_emojis = [e for msg in emojis_list for e in msg if e]
            if not all_emojis:
                return None

            counts = Counter(all_emojis)
            total_with_emoji = sum(1 for msg in emojis_list if msg)
            frequencies = np.array(list(counts.values()))

            # === Fit power-law using powerlaw package ===
            import powerlaw

            fit = powerlaw.Fit(
                frequencies,
                discrete=True,
                xmin=(min_count_for_tail, max_xmin),
                estimate_discrete=True,
            )

            # Extract fit results
            alpha = float(fit.power_law.alpha)
            xmin = int(fit.power_law.xmin)
            D = float(fit.power_law.D)                     # K-S distance

            # Compute log-likelihood on the tail (≥ xmin)
            tail_data = fit.data[fit.data >= xmin]
            loglikelihood = float(np.sum(fit.power_law.loglikelihoods(tail_data)))

            n_tail = int(fit.n_tail)

            # Model comparisons (this is the real proof)
            R_exp, p_exp = fit.distribution_compare('power_law', 'exponential')
            R_log, p_log = fit.distribution_compare('power_law', 'lognormal')

            # Top 10 emojis
            top_10 = [emoji for emoji, _ in counts.most_common(10)]

            result = PowerLawAnalysisResult(
                fit=PowerLawFitResult(
                    alpha=alpha,
                    xmin=xmin,
                    D=D,
                    loglikelihood=loglikelihood,
                    n_tail=n_tail,
                ),
                comparison=ModelComparisonResult(
                    vs_exponential=float(p_exp),
                    vs_lognormal=float(p_log),
                    R=float(R_exp),
                ),
                n_observations=len(counts),
                total_messages_with_emoji=total_with_emoji,
                top_10_emojis=top_10,
            )

            logger.success(
                f"Power-law analysis: α={alpha:.2f}, xmin={xmin}, "
                f"K-S D={D:.3f}, logL={loglikelihood:.1f}, n_tail={n_tail}"
            )
            if result.is_power_law:
                logger.success(
                    "Power-law is the best model (beats exponential & lognormal, p < 0.05)"
                )
            else:
                logger.info(
                    "Power-law fits the tail but not definitively better than alternatives."
                )

            return result

        except Exception as e:
            logger.exception(f"analyze_emoji_distribution_power_law failed: {e}")
            return None
    
    def build_visual_distribution(self, df: pd.DataFrame) -> DistributionPlotData | None:
        df_emoji = self._handle_empty_df(df, "build_visual_distribution")
        if df_emoji.empty:
            return None

        try:
            # Parse from enriched column
            emojis_list = self.data_editor.parse_emojis(df_emoji)
            if not emojis_list.any():
                return None

            # Flatten: each emoji is already a single base character
            all_emojis = [e for msg in emojis_list for e in msg if e]

            if not all_emojis:
                return None

            # Count
            counts = Counter(all_emojis)
            total_with_emoji = sum(1 for msg in emojis_list if msg)

            df_counts = pd.DataFrame(
                counts.items(), columns=["emoji", "count_once"]
            ).sort_values("count_once", ascending=False)

            df_counts["percent_once"] = (df_counts["count_once"] / total_with_emoji) * 100

            # Safe metadata
            df_counts["unicode_code"] = df_counts["emoji"].apply(
                lambda x: " ".join(f"U+{ord(c):04X}" for c in x)
            )
            df_counts["unicode_name"] = df_counts["emoji"].apply(
                lambda x: demojize(x, language="en")
            )

            df_counts = df_counts[
                ["emoji", "count_once", "percent_once", "unicode_name", "unicode_code"]
            ].reset_index(drop=True)

            logger.success(f"Emoji distribution: {len(df_counts)} unique from {total_with_emoji} messages")
            return DistributionPlotData(emoji_counts_df=df_counts)

        except Exception as e:
            logger.exception(f"build_visual_distribution failed: {e}")
            return None


    # === 4. Relationships (Script4) ===
    def build_visual_relationships(self, df_groups: pd.DataFrame) -> RelationshipsPlotData | None:
        """
        Build relationships plot features: avg words, punctuation, message count per author-group.

        Args:
            df_groups: DataFrame with group, author, word/punct counts.

        Returns:
            RelationshipsPlotData or None.

        Raises:
            Exception: If aggregation fails.
        """
        df = self._handle_empty_df(df_groups, "build_visual_relationships")
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
                logger.error(f"Relationships plot missing required input columns: {required_cols}")
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
                f"Relationships feature table built – {len(agg_df)} author-group rows"
            )
            return RelationshipsPlotData(feature_df=agg_df)

        except Exception as e:
            logger.exception(f"build_visual_relationships failed: {e}")
            return None


    # === 5. Multi Dimensions (Script5) ===
    def build_visual_multi_dimensions(
        self,
        df: pd.DataFrame,
        settings: dict,
    ) -> MultiDimPlotData | None:
        """
        Build aggregated style features, embeddings, PCA/t-SNE coordinates, and clustering.

        Args:
            df: Enriched DataFrame with message-level features.
            settings: Dict with keys from Script5ConfigKeys (PLOT_TYPE, BY_GROUP, etc.).

        Returns:
            MultiDimPlotData with:
                - agg_df containing PCA/t-SNE coords and cluster labels
                - style_cols used for modeling
                - model_used (embedding name)
                - plot_type ("pca" | "tsne" | "both")
            Returns None on failure.

        Raises:
            Exception: If any step fails (logged).
        """
        df = self._handle_empty_df(df, "build_visual_multi_dimensions")
        if df.empty:
            return None

        try:
            # === Rename columns to standard names ===
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

            # === Isolate AvT into temp group ===
            df[Columns.WHATSAPP_GROUP_TEMP] = df[Columns.WHATSAPP_GROUP.value]
            df.loc[df[Columns.AUTHOR.value] == Groups.AVT, Columns.WHATSAPP_GROUP_TEMP] = Groups.AVT

            # === Aggregation keys ===
            group_cols = [Columns.AUTHOR.value, Columns.YEAR.value]
            full_group_cols = [*group_cols, Columns.WHATSAPP_GROUP_TEMP]

            # === Aggregation dictionary ===
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

            # === Compute pct_replies ===
            prev = df[Columns.PREVIOUS_AUTHOR.value]
            curr = df[Columns.AUTHOR.value]
            reply_mask = prev.notna() & (prev != curr)
            group_key = df[full_group_cols].apply(tuple, axis=1)
            reply_pct = reply_mask.groupby(group_key).mean()
            reply_df = reply_pct.reset_index(name="pct_replies")
            reply_df[full_group_cols] = pd.DataFrame(reply_df["index"].tolist(), columns=full_group_cols)
            reply_df = reply_df.drop(columns="index")
            agg = agg.merge(reply_df, on=full_group_cols, how="left").fillna(0)

            # === Style features for modeling ===
            style_cols = [
                "msg_count", "words_total", "emojis_total", "punct_total", "caps_total",
                "pics_total", "links_total", "attachments_total", "mean_response", "std_response",
                "pct_ends_emoji", "pct_ends_punct", "pct_has_capitals", "avg_word_len",
                "mean_chat_len", "mean_caps_ratio", "pct_starts_capital", "mean_pct_emojis",
                "pct_starts_emoji", "mean_pct_punct", "pct_has_question", "pct_ends_question",
                "numbers_total", "pct_was_deleted", "pct_replies"
            ]
            X = agg[style_cols].copy()

            # === Embedding settings ===
            use_emb = settings.get(Script5ConfigKeys.USE_EMBEDDINGS, False)
            hybrid = settings.get(Script5ConfigKeys.HYBRID_FEATURES, True)
            model_id = settings.get(Script5ConfigKeys.EMBEDDING_MODEL, EmbeddingModel.MPNET)

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

            # === Resolve plot_type ===
            plot_type = settings.get(Script5ConfigKeys.PLOT_TYPE, "tsne").lower()
            if plot_type not in {"pca", "tsne", "both"}:
                logger.warning(f"Invalid PLOT_TYPE '{plot_type}', defaulting to 'tsne'")
                plot_type = "tsne"

            # === 1. PCA (always computed – needed for t-SNE and pure PCA mode) ===
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            X_scaled = StandardScaler().fit_transform(X)

            n_features = X.shape[1]
            n_components = min(50, n_features)
            logger.info(f"PCA using {n_components} components (max available: {n_features})")

            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X_scaled)                     # (n_samples, n_components)

            # === 2. t-SNE (only when required) ===
            tsne_coords = None
            if plot_type in {"tsne", "both"}:
                from sklearn.manifold import TSNE
                perplexity = min(TSNE_PERPLEXITY_MAX, len(X) - 1)
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                tsne_coords = tsne.fit_transform(X_pca)             # (n_samples, 2)

            # === 3. Write coordinates to agg ===
            if plot_type in {"pca", "both"}:
                agg["pca_1"] = X_pca[:, 0]
                agg["pca_2"] = X_pca[:, 1]

            if plot_type in {"tsne", "both"}:
                agg["tsne_x"] = tsne_coords[:, 0]
                agg["tsne_y"] = tsne_coords[:, 1]

            # === Clustering (on scaled style features) ===
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, metric='euclidean')
            agg["cluster"] = clusterer.fit_predict(X_scaled)

            # === Return enriched model ===
            result = MultiDimPlotData(
                agg_df=agg,
                style_cols=style_cols,
                model_used=model_name,
                by_year=True,
                plot_type=plot_type,
            )
            logger.success(f"MultiDimPlotData built: {len(agg)} rows, {len(style_cols)} features, plot_type={plot_type}")
            return result

        except Exception as e:
            logger.exception(f"build_visual_multi_dimensions failed: {e}")
            return None


# === Public API ===
__all__ = [
    "CategoryPlotData",
    "TimePlotData",
    "DistributionPlotData",
    "RelationshipsPlotData",
    "MultiDimPlotData",
    "MultiDimPlotSettings",
    "SeasonalityEvidence",
    "ComputeSeasonalityEvidence",
    "build_visual_time",
    "RunMode",
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

