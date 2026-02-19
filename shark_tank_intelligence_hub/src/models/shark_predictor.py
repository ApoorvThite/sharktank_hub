"""XGBoost multi-label shark investment classifier."""

import logging
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class SharkPredictor:
    """Multi-label classifier that predicts which sharks will invest in a startup.

    Uses an XGBoost base estimator wrapped in ``MultiOutputClassifier`` so that
    one binary classifier is trained per shark.

    Attributes
    ----------
    model : MultiOutputClassifier or None
        Trained multi-label model. ``None`` until :meth:`train` is called.
    scaler : StandardScaler
        Feature scaler fitted during training.
    feature_columns : List[str]
        Numeric feature columns used for training.
    shark_columns : List[str]
        Target label columns (one per shark).
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.shark_columns: List[str] = []

    def prepare_data(
        self, df: pd.DataFrame, shark_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract feature matrix and label matrix from the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Full cleaned dataset.
        shark_cols : List[str]
            Column names representing each shark's investment decision (0/1).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            ``(X, y)`` where ``X`` is the numeric feature matrix and
            ``y`` is the binary label matrix.
        """
        self.shark_columns = shark_cols
        feature_cols = [col for col in df.columns if col not in shark_cols]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = numeric_cols

        X = df[self.feature_columns].fillna(0)
        y = df[shark_cols].fillna(0).astype(int)
        logger.info("Prepared data: %d samples, %d features, %d labels", len(X), len(numeric_cols), len(shark_cols))
        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Train the multi-label XGBoost classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.DataFrame
            Binary label matrix (one column per shark).
        test_size : float
            Fraction of data to hold out for evaluation.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing ``train_score``, ``test_score``,
            ``X_test``, and ``y_test``.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        base_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric='logloss',
        )
        self.model = MultiOutputClassifier(base_model)
        self.model.fit(X_train_scaled, y_train)

        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        logger.info("Training complete. Train score: %.4f | Test score: %.4f", train_score, test_score)

        return {
            'train_score': train_score,
            'test_score': test_score,
            'X_test': X_test_scaled,
            'y_test': y_test,
        }

    def predict(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate predictions and class probabilities for new samples.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the same columns used during training.

        Returns
        -------
        Tuple[np.ndarray, List[np.ndarray]]
            ``(predictions, probabilities)`` where ``predictions`` is a
            binary array of shape ``(n_samples, n_sharks)`` and
            ``probabilities`` is a list of probability arrays per shark.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        X_scaled = self.scaler.transform(X[self.feature_columns].fillna(0))
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        return predictions, probabilities

    def predict_sharks(self, startup_features: pd.DataFrame) -> List[Dict[str, Any]]:
        """Return per-shark investment predictions sorted by probability.

        Parameters
        ----------
        startup_features : pd.DataFrame
            Single-row DataFrame with startup feature values.

        Returns
        -------
        List[Dict[str, Any]]
            List of dicts with keys ``shark``, ``will_invest``, and
            ``probability``, sorted descending by probability.
        """
        predictions, probabilities = self.predict(startup_features)
        results = []
        for i, shark in enumerate(self.shark_columns):
            prob = probabilities[i][0][1] if len(probabilities[i][0]) > 1 else 0.0
            results.append({
                'shark': shark,
                'will_invest': bool(predictions[0][i]),
                'probability': prob,
            })
        return sorted(results, key=lambda x: x['probability'], reverse=True)

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
            'shark_columns': self.shark_columns,
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
        self.shark_columns = model_data['shark_columns']
        logger.info("Model loaded from %s", filepath)
