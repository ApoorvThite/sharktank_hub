"""Data cleaning pipeline for the Shark Tank India Intelligence Hub."""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Applies a sequential cleaning pipeline to the raw Shark Tank dataset.

    The pipeline runs in this order:
    1. Log missing-value counts.
    2. Standardize column names (lowercase, underscores).
    3. Coerce numeric columns to proper dtypes.
    4. Strip whitespace from string columns.
    5. Remove exact duplicate rows.
    """

    def __init__(self) -> None:
        self.cleaning_log: List[str] = []

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline on a raw DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data.

        Returns
        -------
        pd.DataFrame
            Cleaned copy of the input data.
        """
        df_clean = df.copy()
        df_clean = self._handle_missing_values(df_clean)
        df_clean = self._standardize_columns(df_clean)
        df_clean = self._clean_numeric_fields(df_clean)
        df_clean = self._clean_categorical_fields(df_clean)
        df_clean = self._remove_duplicates(df_clean)
        logger.info("Cleaning complete. Final shape: %s", df_clean.shape)
        return df_clean

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log the current missing-value count (imputation handled downstream)."""
        total_missing = int(df.isnull().sum().sum())
        msg = f"Missing values before cleaning: {total_missing}"
        self.cleaning_log.append(msg)
        logger.info(msg)
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to snake_case lowercase."""
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False)
        return df

    def _clean_numeric_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce all numeric-typed columns to proper numeric dtype."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _clean_categorical_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing whitespace from all string columns."""
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            df[col] = df[col].str.strip()
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop exact duplicate rows and log how many were removed."""
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        if removed > 0:
            msg = f"Removed {removed} duplicate rows"
            self.cleaning_log.append(msg)
            logger.info(msg)
        return df

    def get_cleaning_report(self) -> str:
        """Return a human-readable summary of all cleaning actions taken.

        Returns
        -------
        str
            Newline-separated log of cleaning steps.
        """
        return "\n".join(self.cleaning_log)
