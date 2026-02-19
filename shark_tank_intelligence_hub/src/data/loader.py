"""Data loading utilities for the Shark Tank India Intelligence Hub."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and saving of raw and processed datasets.

    Parameters
    ----------
    data_dir : str
        Root directory for all data files. Defaults to ``'data'``.
    """

    def __init__(self, data_dir: str = 'data') -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'

    def load_raw_data(self, filename: str = 'shark_tank_india.csv') -> pd.DataFrame:
        """Load the raw Shark Tank India CSV dataset.

        Parameters
        ----------
        filename : str
            Name of the CSV file inside ``data/raw/``.

        Returns
        -------
        pd.DataFrame
            Raw dataset.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at the expected path.
        """
        filepath = self.raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Raw data file not found: {filepath}")
        logger.info("Loading raw data from %s", filepath)
        return pd.read_csv(filepath)

    def load_processed_data(self, filename: str = 'processed_data.csv') -> pd.DataFrame:
        """Load a processed/cleaned dataset.

        Parameters
        ----------
        filename : str
            Name of the CSV file inside ``data/processed/``.

        Returns
        -------
        pd.DataFrame
            Processed dataset.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at the expected path.
        """
        filepath = self.processed_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Processed data file not found: {filepath}")
        logger.info("Loading processed data from %s", filepath)
        return pd.read_csv(filepath)

    def save_processed_data(self, df: pd.DataFrame, filename: str = 'processed_data.csv') -> None:
        """Persist a DataFrame to the processed data directory.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save.
        filename : str
            Output filename inside ``data/processed/``.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.processed_dir / filename
        df.to_csv(filepath, index=False)
        logger.info("Saved processed data to %s (%d rows)", filepath, len(df))

    def load_industry_benchmarks(self) -> Optional[pd.DataFrame]:
        """Load pre-computed industry benchmark statistics.

        Returns
        -------
        pd.DataFrame or None
            Benchmark data, or ``None`` if the file is not found.
        """
        filepath = self.data_dir / 'industry_benchmarks' / 'benchmarks.csv'
        if filepath.exists():
            logger.info("Loading industry benchmarks from %s", filepath)
            return pd.read_csv(filepath)
        logger.warning("Industry benchmarks file not found at %s", filepath)
        return None

    def load_shark_profiles(self) -> Optional[pd.DataFrame]:
        """Load pre-computed shark profile statistics.

        Returns
        -------
        pd.DataFrame or None
            Shark profiles, or ``None`` if the file is not found.
        """
        filepath = self.data_dir / 'shark_profiles' / 'profiles.csv'
        if filepath.exists():
            logger.info("Loading shark profiles from %s", filepath)
            return pd.read_csv(filepath)
        logger.warning("Shark profiles file not found at %s", filepath)
        return None
