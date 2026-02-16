import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ModelExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def plot_feature_importance(self, top_n=20, figsize=(10, 8)):
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=figsize)
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            return plt.gcf()
        else:
            print("Model does not have feature_importances_ attribute")
            return None
    
    def get_prediction_explanation(self, X, prediction_idx=0):
        explanation = {
            'prediction': None,
            'top_features': []
        }
        
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(X)
            explanation['prediction'] = prediction[prediction_idx]
        
        if hasattr(self.model, 'feature_importances_'):
            feature_values = X.iloc[prediction_idx] if isinstance(X, pd.DataFrame) else X[prediction_idx]
            
            feature_contributions = []
            for i, (feat_name, feat_val) in enumerate(zip(self.feature_names, feature_values)):
                contribution = {
                    'feature': feat_name,
                    'value': feat_val,
                    'importance': self.model.feature_importances_[i]
                }
                feature_contributions.append(contribution)
            
            explanation['top_features'] = sorted(
                feature_contributions, 
                key=lambda x: x['importance'], 
                reverse=True
            )[:10]
        
        return explanation
    
    def generate_report(self, X_test, y_test, predictions):
        report = {
            'model_type': type(self.model).__name__,
            'num_features': len(self.feature_names),
            'num_samples': len(X_test),
            'top_features': []
        }
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            report['top_features'] = importance_df.head(10).to_dict('records')
        
        return report
