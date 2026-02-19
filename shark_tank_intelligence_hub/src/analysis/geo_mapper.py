"""Geographic analysis of Shark Tank India investment patterns by state/region."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class GeoMapper:
    """Computes location-level investment statistics and identifies startup hubs.

    Attributes
    ----------
    location_stats : Dict[str, Dict[str, Any]]
        Mapping of state/location name to its computed statistics. Populated
        by :meth:`analyze_geographic_patterns`.
    """

    def __init__(self) -> None:
        self.location_stats: Dict[str, Dict[str, Any]] = {}

    def analyze_geographic_patterns(
        self, df: pd.DataFrame, location_col: str = 'state'
    ) -> Dict[str, Dict[str, Any]]:
        """Compute summary statistics for every state/location in the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned deals dataset.
        location_col : str
            Column name containing state or city labels.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Per-location statistics: pitch count, deal count, success rate,
            average investment, total investment, and average equity.
        """
        for location in df[location_col].dropna().unique():
            subset = df[df[location_col] == location]
            n = len(subset)

            deals_made = subset['deal_accepted'].sum() if 'deal_accepted' in subset else 0
            self.location_stats[location] = {
                'total_pitches': n,
                'deals_made': int(deals_made),
                'success_rate': (deals_made / n * 100) if n > 0 else 0.0,
                'avg_investment': subset['amount_invested'].mean() if 'amount_invested' in subset else 0.0,
                'total_investment': subset['amount_invested'].sum() if 'amount_invested' in subset else 0.0,
                'avg_equity': subset['equity_taken'].mean() if 'equity_taken' in subset else 0.0,
            }

        logger.info("Geographic analysis complete: %d locations.", len(self.location_stats))
        return self.location_stats

    def get_top_locations(
        self, metric: str = 'total_pitches', top_n: int = 10
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Return the top-N locations ranked by a given metric.

        Parameters
        ----------
        metric : str
            Metric key to rank by (e.g. ``'success_rate'``, ``'total_investment'``).
        top_n : int
            Number of locations to return.

        Returns
        -------
        List[Tuple[str, Dict[str, Any]]]
            ``(location_name, stats)`` pairs sorted descending by ``metric``.
        """
        return sorted(
            self.location_stats.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True,
        )[:top_n]

    def get_location_benchmarks(self, location: str) -> Optional[Dict[str, Any]]:
        """Return benchmark statistics for a single location.

        Parameters
        ----------
        location : str
            State or city name as it appears in the dataset.

        Returns
        -------
        Dict[str, Any] or None
            Statistics dict, or ``None`` if the location was not found.
        """
        if location not in self.location_stats:
            logger.warning("Location '%s' not found in stats.", location)
            return None
        return self.location_stats[location]

    def analyze_regional_trends(
        self,
        df: pd.DataFrame,
        location_col: str = 'state',
        season_col: str = 'season',
    ) -> Dict[str, Any]:
        """Compute per-season investment trends for each location.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned deals dataset.
        location_col : str
            Column name for state/location labels.
        season_col : str
            Column name for the season number.

        Returns
        -------
        Dict[str, Any]
            Nested dict: ``{location: {season: {metric: value}}}``.
        """
        trends: Dict[str, Any] = {}
        agg_cols = {c: ('sum' if c == 'deal_accepted' else 'sum')
                    for c in ['deal_accepted', 'amount_invested']
                    if c in df.columns}

        for location in df[location_col].dropna().unique():
            subset = df[df[location_col] == location]
            if agg_cols:
                trends[location] = subset.groupby(season_col).agg(agg_cols).to_dict('index')
            else:
                trends[location] = {}

        return trends

    def identify_startup_hubs(
        self, min_pitches: int = 10
    ) -> List[Dict[str, Any]]:
        """Identify locations with significant startup activity.

        Parameters
        ----------
        min_pitches : int
            Minimum number of pitches required to qualify as a hub.

        Returns
        -------
        List[Dict[str, Any]]
            Hub locations sorted descending by ``total_pitches``, each with
            keys ``location``, ``total_pitches``, ``success_rate``, and
            ``total_investment``.
        """
        hubs = [
            {
                'location': loc,
                'total_pitches': stats['total_pitches'],
                'success_rate': stats['success_rate'],
                'total_investment': stats['total_investment'],
            }
            for loc, stats in self.location_stats.items()
            if stats['total_pitches'] >= min_pitches
        ]
        return sorted(hubs, key=lambda x: x['total_pitches'], reverse=True)

    def compare_locations(
        self, location1: str, location2: str
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Side-by-side comparison of two locations across all metrics.

        Parameters
        ----------
        location1 : str
            First location name.
        location2 : str
            Second location name.

        Returns
        -------
        Dict[str, Dict[str, Any]] or None
            Per-metric comparison dict, or ``None`` if either location is missing.
        """
        if location1 not in self.location_stats or location2 not in self.location_stats:
            logger.warning("One or both locations not found: '%s', '%s'", location1, location2)
            return None

        comparison: Dict[str, Dict[str, Any]] = {}
        for metric in self.location_stats[location1]:
            v1 = self.location_stats[location1][metric]
            v2 = self.location_stats[location2][metric]
            comparison[metric] = {
                location1: v1,
                location2: v2,
                'difference': v1 - v2 if isinstance(v1, (int, float)) else None,
            }
        return comparison

    def generate_geographic_report(self) -> pd.DataFrame:
        """Return a DataFrame summarising all location statistics.

        Returns
        -------
        pd.DataFrame
            One row per location, sorted by ``total_pitches`` descending.
        """
        report_df = pd.DataFrame.from_dict(self.location_stats, orient='index')
        return report_df.sort_values('total_pitches', ascending=False)
