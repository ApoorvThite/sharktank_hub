"""Feature importance and prediction explanation utilities."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Provides feature importance plots and per-prediction explanations.

    Parameters
    ----------
    model : Any
        A fitted sklearn-compatible estimator with a ``predict`` method.
    feature_names : List[str]
        Ordered list of feature names matching the model's input.
    """

    def __init__(self, model: Any, feature_names: List[str]) -> None:
        self.model = model
        self.feature_names = feature_names

    def plot_feature_importance(
        self,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Optional[plt.Figure]:
        """Plot a horizontal bar chart of the top-N feature importances.

        Parameters
        ----------
        top_n : int
            Number of top features to display.
        figsize : Tuple[int, int]
            Figure dimensions ``(width, height)`` in inches.

        Returns
        -------
        matplotlib.figure.Figure or None
            The generated figure, or ``None`` if the model does not expose
            ``feature_importances_``.
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not expose feature_importances_. Skipping plot.")
            return None

        importance_df = (
            pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_,
            })
            .sort_values('importance', ascending=False)
            .head(top_n)
        )

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        fig.tight_layout()
        logger.info("Feature importance plot generated for top %d features.", top_n)
        return fig

    def get_prediction_explanation(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        prediction_idx: int = 0,
    ) -> Dict[str, Any]:
        """Explain a single prediction using feature importances as a proxy.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix (may contain multiple rows).
        prediction_idx : int
            Row index of the sample to explain.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - ``prediction``: the model's raw prediction for the sample.
            - ``top_features``: list of the 10 most important features with
              their values and global importances.
        """
        explanation: Dict[str, Any] = {'prediction': None, 'top_features': []}

        if hasattr(self.model, 'predict'):
            explanation['prediction'] = self.model.predict(X)[prediction_idx]

        if hasattr(self.model, 'feature_importances_'):
            feature_values = (
                X.iloc[prediction_idx] if isinstance(X, pd.DataFrame) else X[prediction_idx]
            )
            contributions = [
                {
                    'feature': feat_name,
                    'value': feat_val,
                    'importance': self.model.feature_importances_[i],
                }
                for i, (feat_name, feat_val) in enumerate(
                    zip(self.feature_names, feature_values)
                )
            ]
            explanation['top_features'] = sorted(
                contributions, key=lambda x: x['importance'], reverse=True
            )[:10]

        return explanation

    def generate_report(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        predictions: np.ndarray,
    ) -> Dict[str, Any]:
        """Generate a summary report of model metadata and top feature importances.

        Parameters
        ----------
        X_test : pd.DataFrame or np.ndarray
            Test feature matrix.
        y_test : pd.Series or np.ndarray
            True target values.
        predictions : np.ndarray
            Model predictions on the test set.

        Returns
        -------
        Dict[str, Any]
            Report containing ``model_type``, ``num_features``,
            ``num_samples``, and ``top_features``.
        """
        report: Dict[str, Any] = {
            'model_type': type(self.model).__name__,
            'num_features': len(self.feature_names),
            'num_samples': len(X_test),
            'top_features': [],
        }

        if hasattr(self.model, 'feature_importances_'):
            importance_df = (
                pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_,
                })
                .sort_values('importance', ascending=False)
            )
            report['top_features'] = importance_df.head(10).to_dict('records')

        logger.info("Report generated for %s with %d test samples.", report['model_type'], report['num_samples'])
        return report
