"""Feature engineering pipeline for the Shark Tank India Intelligence Hub."""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates derived features from the cleaned Shark Tank dataset.

    Feature groups created:
    - **Deal features**: implied valuation, deal success flag.
    - **Valuation features**: asked valuation, valuation gap (absolute & %).
    - **Shark features**: number of sharks involved per deal.
    - **Temporal features**: numeric season and episode numbers.
    - **Industry features**: label-encoded industry category.
    - **Geographic features**: label-encoded state/location.
    """

    def __init__(self) -> None:
        self.feature_names: List[str] = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned dataset.

        Returns
        -------
        pd.DataFrame
            Dataset with all engineered features appended.
        """
        df_features = df.copy()
        df_features = self._create_deal_features(df_features)
        df_features = self._create_valuation_features(df_features)
        df_features = self._create_shark_features(df_features)
        df_features = self._create_temporal_features(df_features)
        df_features = self._create_industry_features(df_features)
        df_features = self._create_geographic_features(df_features)
        logger.info("Feature engineering complete. %d new features created.", len(self.feature_names))
        return df_features

    def _create_deal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create deal outcome and implied-valuation features."""
        if 'deal_accepted' in df.columns:
            df['deal_success'] = df['deal_accepted'].astype(int)

        if 'amount_invested' in df.columns and 'equity_taken' in df.columns:
            df['implied_valuation'] = np.where(
                df['equity_taken'] > 0,
                (df['amount_invested'] / df['equity_taken']) * 100,
                np.nan,
            )
            self.feature_names.append('implied_valuation')
        return df

    def _create_valuation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create asked-valuation and valuation-gap features."""
        if 'asked_amount' in df.columns and 'asked_equity' in df.columns:
            df['asked_valuation'] = np.where(
                df['asked_equity'] > 0,
                (df['asked_amount'] / df['asked_equity']) * 100,
                np.nan,
            )
            self.feature_names.append('asked_valuation')

        if 'implied_valuation' in df.columns and 'asked_valuation' in df.columns:
            df['valuation_gap'] = df['asked_valuation'] - df['implied_valuation']
            df['valuation_gap_pct'] = (df['valuation_gap'] / df['asked_valuation']) * 100
            self.feature_names.extend(['valuation_gap', 'valuation_gap_pct'])
        return df

    def _create_shark_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a feature counting how many sharks participated in each deal."""
        shark_cols = [col for col in df.columns if 'shark' in col.lower()]
        if shark_cols:
            df['num_sharks_involved'] = df[shark_cols].sum(axis=1)
            self.feature_names.append('num_sharks_involved')
        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create numeric season and episode features."""
        if 'season' in df.columns:
            df['season_num'] = df['season'].astype(int)
            self.feature_names.append('season_num')

        if 'episode' in df.columns:
            df['episode_num'] = df['episode'].astype(int)
            self.feature_names.append('episode_num')
        return df

    def _create_industry_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode the industry category column."""
        if 'industry' in df.columns:
            df['industry_encoded'] = pd.Categorical(df['industry']).codes
            self.feature_names.append('industry_encoded')
        return df

    def _create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode the state/location column."""
        if 'state' in df.columns or 'location' in df.columns:
            location_col = 'state' if 'state' in df.columns else 'location'
            df['location_encoded'] = pd.Categorical(df[location_col]).codes
            self.feature_names.append('location_encoded')
        return df

    def get_feature_list(self) -> List[str]:
        """Return the list of all engineered feature names.

        Returns
        -------
        List[str]
            Names of features created by this pipeline.
        """
        return self.feature_names
