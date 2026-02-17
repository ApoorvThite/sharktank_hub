"""
SHARK TANK PREDICTOR - Prediction Function
Complete prediction pipeline for new startups
"""

import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

class SharkTankPredictor:
    """
    Complete prediction system for Shark Tank India startups
    """
    
    def __init__(self):
        """Load all trained models"""
        print("Loading models...")
        
        # Load binary classifier
        with open('models/shark_predictor_tuned.pkl', 'rb') as f:
            self.binary_model = pickle.load(f)
        
        # Load multi-label classifiers
        with open('models/shark_multilabel_models.pkl', 'rb') as f:
            self.shark_models = pickle.load(f)
        
        # Load regression model
        with open('models/equity_predictor_regression.pkl', 'rb') as f:
            self.regression_model = pickle.load(f)
        
        # Load feature names
        self.feature_names = pd.read_csv('data/processed/feature_list.csv')['Feature'].tolist()
        
        print(f"✅ Models loaded successfully")
        print(f"   Features required: {len(self.feature_names)}")
    
    def predict(self, startup_features):
        """
        Predict deal outcome for a new startup
        
        Parameters:
        -----------
        startup_features : dict or pd.DataFrame
            Dictionary or DataFrame with startup features
        
        Returns:
        --------
        dict with predictions:
            - offer_probability: float (0-1)
            - will_get_offer: bool
            - shark_probabilities: dict {shark: probability}
            - recommended_sharks: list of sharks sorted by probability
            - expected_equity_dilution: float (percentage)
            - prediction_confidence: str (High/Medium/Low)
        """
        
        # Convert to DataFrame if dict
        if isinstance(startup_features, dict):
            startup_features = pd.DataFrame([startup_features])
        
        # Ensure all required features are present
        for feat in self.feature_names:
            if feat not in startup_features.columns:
                startup_features[feat] = 0
        
        # Select and order features
        X = startup_features[self.feature_names]
        
        # 1. Predict offer probability
        offer_prob = self.binary_model.predict_proba(X)[0][1]
        will_get_offer = offer_prob > 0.5
        
        # 2. Predict individual shark probabilities
        shark_probs = {}
        for shark, model in self.shark_models.items():
            shark_probs[shark] = model.predict_proba(X)[0][1]
        
        # Sort sharks by probability
        recommended_sharks = sorted(shark_probs.items(), key=lambda x: x[1], reverse=True)
        
        # 3. Predict equity dilution
        equity_dilution = self.regression_model.predict(X)[0]
        
        # 4. Determine confidence
        if offer_prob > 0.8 or offer_prob < 0.2:
            confidence = "High"
        elif offer_prob > 0.65 or offer_prob < 0.35:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            'offer_probability': round(offer_prob, 4),
            'will_get_offer': will_get_offer,
            'shark_probabilities': {k: round(v, 4) for k, v in shark_probs.items()},
            'recommended_sharks': [(shark, round(prob, 4)) for shark, prob in recommended_sharks[:3]],
            'expected_equity_dilution': round(equity_dilution, 2),
            'prediction_confidence': confidence
        }
    
    def explain_prediction(self, startup_features, startup_name="Startup"):
        """
        Provide detailed explanation of prediction
        """
        prediction = self.predict(startup_features)
        
        print("="*80)
        print(f"🦈 SHARK TANK PREDICTION FOR: {startup_name}")
        print("="*80)
        
        print(f"\n📊 OFFER PREDICTION:")
        print(f"   Probability of Getting Offer: {prediction['offer_probability']*100:.1f}%")
        print(f"   Prediction: {'✅ WILL GET OFFER' if prediction['will_get_offer'] else '❌ NO OFFER'}")
        print(f"   Confidence: {prediction['prediction_confidence']}")
        
        print(f"\n🦈 SHARK RECOMMENDATIONS (Top 3):")
        for i, (shark, prob) in enumerate(prediction['recommended_sharks'], 1):
            print(f"   {i}. {shark.capitalize():10s} - {prob*100:5.1f}% probability")
        
        print(f"\n💰 EQUITY PREDICTION:")
        if prediction['will_get_offer']:
            print(f"   Expected Equity Dilution: {prediction['expected_equity_dilution']:.2f}%")
        else:
            print(f"   N/A (No offer predicted)")
        
        print(f"\n📊 ALL SHARK PROBABILITIES:")
        sorted_sharks = sorted(prediction['shark_probabilities'].items(), 
                              key=lambda x: x[1], reverse=True)
        for shark, prob in sorted_sharks:
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"   {shark.capitalize():10s} [{bar}] {prob*100:5.1f}%")
        
        print("="*80)
        
        return prediction


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("🦈 SHARK TANK PREDICTOR - DEMO")
    print("="*80)
    
    # Initialize predictor
    predictor = SharkTankPredictor()
    
    # Load a sample from test set
    print("\n📊 Loading sample startup from test set...")
    X_test = pd.read_csv('data/processed/X_test.csv')
    
    # Example 1: High-revenue tech startup
    print("\n" + "="*80)
    print("EXAMPLE 1: High-Revenue Tech Startup")
    print("="*80)
    sample_1 = X_test.iloc[0:1]
    prediction_1 = predictor.explain_prediction(sample_1, "HealthTech Startup")
    
    # Example 2: Pre-revenue food startup
    print("\n" + "="*80)
    print("EXAMPLE 2: Different Startup Profile")
    print("="*80)
    sample_2 = X_test.iloc[10:11]
    prediction_2 = predictor.explain_prediction(sample_2, "Food & Beverage Startup")
    
    # Example 3: Mid-revenue fashion startup
    print("\n" + "="*80)
    print("EXAMPLE 3: Another Startup Profile")
    print("="*80)
    sample_3 = X_test.iloc[20:21]
    prediction_3 = predictor.explain_prediction(sample_3, "Fashion Startup")
    
    print("\n✅ Prediction demo complete!")
    print("="*80)
