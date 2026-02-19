"""Deal structure classification and analysis for Shark Tank India deals."""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DealDecoder:
    """Classifies and analyses the structure of Shark Tank India deals.

    Deal structures recognised:
    - **Equity only** – no debt or royalty component.
    - **Equity with debt** – includes a debt tranche.
    - **Equity with royalty** – includes a revenue-share royalty.
    - **Complex** – both debt and royalty present.

    Attributes
    ----------
    deal_patterns : Dict[str, Any]
        Stores computed pattern summaries after analysis methods are called.
    """

    def __init__(self) -> None:
        self.deal_patterns: Dict[str, Any] = {}

    def analyze_deal_structures(self, df: pd.DataFrame) -> Dict[str, int]:
        """Count deals by structure type across the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Deals dataset. Expected columns: ``debt_amount``, ``royalty``.

        Returns
        -------
        Dict[str, int]
            Counts for ``equity_only``, ``equity_with_debt``,
            ``equity_with_royalty``, and ``complex_deals``.
        """
        deal_types: Dict[str, int] = {
            'equity_only': 0,
            'equity_with_debt': 0,
            'equity_with_royalty': 0,
            'complex_deals': 0,
        }

        debt_col = 'debt_amount' if 'debt_amount' in df.columns else None
        royalty_col = 'royalty' if 'royalty' in df.columns else None

        for _, row in df.iterrows():
            has_debt = bool(row.get(debt_col, 0) > 0) if debt_col else False
            has_royalty = bool(row.get(royalty_col, 0) > 0) if royalty_col else False

            if has_debt and has_royalty:
                deal_types['complex_deals'] += 1
            elif has_debt:
                deal_types['equity_with_debt'] += 1
            elif has_royalty:
                deal_types['equity_with_royalty'] += 1
            else:
                deal_types['equity_only'] += 1

        self.deal_patterns['structure_distribution'] = deal_types
        logger.info("Deal structure analysis complete: %s", deal_types)
        return deal_types

    def analyze_debt_terms(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Summarise debt terms across all debt-bearing deals.

        Parameters
        ----------
        df : pd.DataFrame
            Deals dataset with ``debt_amount`` and ``interest_rate`` columns.

        Returns
        -------
        Dict[str, float] or None
            Summary statistics, or ``None`` if no debt deals exist.
        """
        if 'debt_amount' not in df.columns:
            logger.warning("'debt_amount' column not found. Skipping debt analysis.")
            return None

        debt_deals = df[df['debt_amount'] > 0]
        if debt_deals.empty:
            return None

        return {
            'total_debt_deals': len(debt_deals),
            'avg_debt_amount': float(debt_deals['debt_amount'].mean()),
            'avg_interest_rate': float(debt_deals['interest_rate'].mean()) if 'interest_rate' in debt_deals else 0.0,
            'total_debt_issued': float(debt_deals['debt_amount'].sum()),
        }

    def analyze_royalty_terms(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Summarise royalty terms across all royalty-bearing deals.

        Parameters
        ----------
        df : pd.DataFrame
            Deals dataset with ``royalty`` and optionally ``royalty_duration``.

        Returns
        -------
        Dict[str, float] or None
            Summary statistics, or ``None`` if no royalty deals exist.
        """
        if 'royalty' not in df.columns:
            logger.warning("'royalty' column not found. Skipping royalty analysis.")
            return None

        royalty_deals = df[df['royalty'] > 0]
        if royalty_deals.empty:
            return None

        return {
            'total_royalty_deals': len(royalty_deals),
            'avg_royalty_percentage': float(royalty_deals['royalty'].mean()),
            'avg_royalty_duration': (
                float(royalty_deals['royalty_duration'].mean())
                if 'royalty_duration' in royalty_deals else 0.0
            ),
        }

    def identify_special_terms(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract rows that contain non-standard deal terms.

        Parameters
        ----------
        df : pd.DataFrame
            Deals dataset.

        Returns
        -------
        List[Dict[str, Any]]
            Each entry has ``pitch_id``, ``startup``, and ``special_terms``
            (a comma-separated description of non-standard components).
        """
        special_terms: List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            terms: List[str] = []
            if row.get('debt_amount', 0) > 0:
                terms.append(f"Debt: ₹{row['debt_amount']}L")
            if row.get('royalty', 0) > 0:
                terms.append(f"Royalty: {row['royalty']}%")
            if row.get('advisory_equity', 0) > 0:
                terms.append(f"Advisory: {row['advisory_equity']}%")

            if terms:
                special_terms.append({
                    'pitch_id': idx,
                    'startup': row.get('startup_name', 'Unknown'),
                    'special_terms': ', '.join(terms),
                })

        logger.info("Identified %d deals with special terms.", len(special_terms))
        return special_terms

    def calculate_effective_cost(
        self,
        amount: float,
        equity: float,
        debt: float = 0.0,
        royalty: float = 0.0,
        royalty_duration: float = 0.0,
    ) -> Dict[str, float]:
        """Compute the total effective cost of a deal structure.

        Parameters
        ----------
        amount : float
            Equity investment amount in ₹ Lakhs.
        equity : float
            Equity percentage given to investors.
        debt : float
            Debt component in ₹ Lakhs (default 0).
        royalty : float
            Royalty percentage on revenue (default 0).
        royalty_duration : float
            Number of years the royalty applies (default 0).

        Returns
        -------
        Dict[str, float]
            Breakdown: ``equity_cost``, ``debt_cost``, ``royalty_cost``,
            and ``total_effective_cost``.
        """
        equity_cost = amount / (equity / 100) if equity > 0 else 0.0
        debt_cost = debt * 1.12  # assumes 12% flat interest
        royalty_cost = (royalty / 100) * royalty_duration * amount if royalty > 0 else 0.0

        return {
            'equity_cost': equity_cost,
            'debt_cost': debt_cost,
            'royalty_cost': royalty_cost,
            'total_effective_cost': equity_cost + debt_cost + royalty_cost,
        }

    def get_deal_complexity_score(self, row: pd.Series) -> int:
        """Assign a complexity score to a single deal row.

        Scoring:
        - Equity component: +1
        - Debt component: +2
        - Royalty component: +2
        - Advisory equity: +1
        - Each additional shark: +1

        Parameters
        ----------
        row : pd.Series
            A single deal row from the dataset.

        Returns
        -------
        int
            Complexity score (higher = more complex deal).
        """
        score = 0
        if row.get('equity_taken', 0) > 0:
            score += 1
        if row.get('debt_amount', 0) > 0:
            score += 2
        if row.get('royalty', 0) > 0:
            score += 2
        if row.get('advisory_equity', 0) > 0:
            score += 1
        score += sum(1 for col in row.index if 'shark' in col.lower() and row[col] == 1)
        return score
