# src/scripts/script7.py
from .base import BaseScript
from src.data_preparation import DataPreparation
from src.plot_manager import PlotManager, DimReductionSettings, PMNoMessageContentSettings

class Script7(BaseScript):
    def __init__(self, file_manager, data_preparation, plot_manager, image_dir, group_authors, df=None, settings=None):
        super().__init__(file_manager, data_preparation=data_preparation, plot_manager=plot_manager, settings=settings, df=df)
        self.image_dir = image_dir
        self.group_authors = group_authors

    def run(self):
        df_filtered = self.data_preparation.df[self.data_preparation.df['whatsapp_group'] != 'tillies'].copy()
        if df_filtered.empty:
            return self.log_error("No data after filtering 'tillies'.")
        group_authors_filtered = {g: a for g, a in self.group_authors.items() if g != 'tillies'}
        if not group_authors_filtered:
            return self.log_error("No groups remain.")
        feature_df = self.data_preparation.build_interaction_features(df_filtered, group_authors_filtered)
        if feature_df is None:
            return self.log_error("Failed to build features.")
        fig_interact, fig_groups = self.plot_manager.build_visual_interactions_2(
            feature_df,
            method='pca',
            settings=DimReductionSettings(),
            nmc_settings=PMNoMessageContentSettings()
        )
        if fig_interact:
            self.save_figure(fig_interact, self.image_dir, "interaction_dynamics_pca")
        if fig_groups:
            self.save_figure(fig_groups, self.image_dir, "interaction_dynamics_groups_pca")