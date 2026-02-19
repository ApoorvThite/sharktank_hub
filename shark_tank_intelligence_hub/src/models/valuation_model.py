"""Random Forest regression model for equity dilution prediction."""

import logging
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ValuationModel:
    """Random Forest regressor that predicts equity dilution for a startup deal.

    Attributes
    ----------
    model : RandomForestRegressor
        Underlying sklearn regressor.
    scaler : StandardScaler
        Feature scaler fitted during training.
    feature_columns : List[str]
        Numeric feature columns used for training.
    """

    def __init__(self) -> None:
        self.model: RandomForestRegressor = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: List[str] = []

    def prepare_data(
        self, df: pd.DataFrame, target_col: str = 'equity_taken'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract feature matrix and target series from the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned dataset containing both features and target.
        target_col : str
            Name of the column to predict (default: ``'equity_taken'``).

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            ``(X, y)`` feature matrix and target series.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        self.feature_columns = numeric_cols

        X = df[self.feature_columns].fillna(0)
        y = df[target_col].fillna(0)
        logger.info("Prepared data: %d samples, %d features, target='%s'", len(X), len(numeric_cols), target_col)
        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Fit the Random Forest regressor and evaluate on a held-out test set.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target values (equity percentage).
        test_size : float
            Fraction of data to hold out for evaluation.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Training metrics: ``train_mae``, ``test_mae``, ``train_r2``,
            ``test_r2``, ``predictions``, and ``actuals``.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'predictions': test_pred,
            'actuals': y_test,
        }
        logger.info(
            "Training complete. Test MAE: %.4f | Test R²: %.4f",
            metrics['test_mae'],
            metrics['test_r2'],
        )
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict equity dilution for new startup samples.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the same columns used during training.

        Returns
        -------
        np.ndarray
            Predicted equity percentages.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        X_scaled = self.scaler.transform(X[self.feature_columns].fillna(0))
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of feature importances sorted descending.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        return (
            pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_,
            })
            .sort_values('importance', ascending=False)
            .reset_index(drop=True)
        )

    def save_model(self, filepath: str) -> None:
        """Serialize the trained model and metadata to disk.

        Parameters
        ----------
        filepath : str
            Destination ``.pkl`` file path.
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info("Model saved to %s", filepath)

    def load_model(self, filepath: str) -> None:
        """Load a previously serialized model from disk.

        Parameters
        ----------
        filepath : str
            Path to the ``.pkl`` file produced by :meth:`save_model`.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        logger.info("Model loaded from %s", filepath)
