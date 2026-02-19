"""Industry-level profiling and benchmarking for Shark Tank India analysis."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class IndustryProfiler:
    """Computes and stores aggregate statistics for each industry category.

    Attributes
    ----------
    industry_stats : Dict[str, Dict[str, Any]]
        Mapping of industry name to its computed statistics. Populated by
        :meth:`analyze_industries`.
    """

    def __init__(self) -> None:
        self.industry_stats: Dict[str, Dict[str, Any]] = {}

    def analyze_industries(
        self, df: pd.DataFrame, industry_col: str = 'industry'
    ) -> Dict[str, Dict[str, Any]]:
        """Compute summary statistics for every industry in the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned deals dataset.
        industry_col : str
            Name of the column containing industry labels.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Per-industry statistics including pitch count, success rate,
            average investment, equity, and valuation.
        """
        for industry in df[industry_col].unique():
            subset = df[df[industry_col] == industry]
            n = len(subset)

            deals_made = subset['deal_accepted'].sum() if 'deal_accepted' in subset else 0
            self.industry_stats[industry] = {
                'total_pitches': n,
                'deals_made': int(deals_made),
                'success_rate': (deals_made / n * 100) if n > 0 else 0.0,
                'avg_investment': subset['amount_invested'].mean() if 'amount_invested' in subset else 0.0,
                'avg_equity': subset['equity_taken'].mean() if 'equity_taken' in subset else 0.0,
                'total_investment': subset['amount_invested'].sum() if 'amount_invested' in subset else 0.0,
                'avg_valuation': subset['implied_valuation'].mean() if 'implied_valuation' in subset else 0.0,
            }

        logger.info("Analyzed %d industries.", len(self.industry_stats))
        return self.industry_stats

    def get_industry_benchmarks(self, industry: str) -> Optional[Dict[str, Any]]:
        """Return benchmark statistics for a single industry.

        Parameters
        ----------
        industry : str
            Industry name as it appears in the dataset.

        Returns
        -------
        Dict[str, Any] or None
            Statistics dict, or ``None`` if the industry was not found.
        """
        if industry not in self.industry_stats:
            logger.warning("Industry '%s' not found in stats.", industry)
            return None
        return self.industry_stats[industry]

    def get_top_industries(
        self, metric: str = 'total_pitches', top_n: int = 10
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Return the top-N industries ranked by a given metric.

        Parameters
        ----------
        metric : str
            Metric key to rank by (e.g. ``'success_rate'``, ``'total_pitches'``).
        top_n : int
            Number of industries to return.

        Returns
        -------
        List[Tuple[str, Dict[str, Any]]]
            ``(industry_name, stats)`` pairs sorted descending by ``metric``.
        """
        return sorted(
            self.industry_stats.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True,
        )[:top_n]

    def compare_industries(
        self, industry1: str, industry2: str
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Side-by-side comparison of two industries across all metrics.

        Parameters
        ----------
        industry1 : str
            First industry name.
        industry2 : str
            Second industry name.

        Returns
        -------
        Dict[str, Dict[str, Any]] or None
            Per-metric comparison dict, or ``None`` if either industry is missing.
        """
        if industry1 not in self.industry_stats or industry2 not in self.industry_stats:
            logger.warning("One or both industries not found: '%s', '%s'", industry1, industry2)
            return None

        comparison: Dict[str, Dict[str, Any]] = {}
        for metric in self.industry_stats[industry1]:
            v1 = self.industry_stats[industry1][metric]
            v2 = self.industry_stats[industry2][metric]
            comparison[metric] = {
                industry1: v1,
                industry2: v2,
                'difference': v1 - v2 if isinstance(v1, (int, float)) else None,
            }
        return comparison

    def get_industry_trends(
        self,
        df: pd.DataFrame,
        industry_col: str = 'industry',
        season_col: str = 'season',
    ) -> Dict[str, Any]:
        """Compute per-season trends for each industry.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned deals dataset.
        industry_col : str
            Column name for industry labels.
        season_col : str
            Column name for the season number.

        Returns
        -------
        Dict[str, Any]
            Nested dict: ``{industry: {season: {metric: value}}}``.
        """
        trends: Dict[str, Any] = {}
        agg_cols = {c: 'sum' if c == 'deal_accepted' else 'mean'
                    for c in ['deal_accepted', 'amount_invested', 'equity_taken']
                    if c in df.columns}

        for industry in df[industry_col].unique():
            subset = df[df[industry_col] == industry]
            if agg_cols:
                trends[industry] = subset.groupby(season_col).agg(agg_cols).to_dict('index')
            else:
                trends[industry] = {}

        return trends

    def generate_industry_report(self) -> pd.DataFrame:
        """Return a DataFrame summarising all industry statistics.

        Returns
        -------
        pd.DataFrame
            One row per industry, sorted by ``total_pitches`` descending.
        """
        report_df = pd.DataFrame.from_dict(self.industry_stats, orient='index')
        return report_df.sort_values('total_pitches', ascending=False)
