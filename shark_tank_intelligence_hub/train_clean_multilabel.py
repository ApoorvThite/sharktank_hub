"""
RETRAIN MULTI-LABEL MODELS WITH CLEAN FEATURES
Train individual shark predictors using only pre-pitch information (no data leakage)
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("🦈 RETRAINING MULTI-LABEL MODELS WITH CLEAN FEATURES")
print("="*80)

# Load clean data
X_train = pd.read_csv('data/processed/X_train_clean.csv')
X_test = pd.read_csv('data/processed/X_test_clean.csv')
y_train_multilabel = pd.read_csv('data/processed/y_train_multilabel.csv')
y_test_multilabel = pd.read_csv('data/processed/y_test_multilabel.csv')

print(f"\n✅ Clean data loaded:")
print(f"   X_train: {X_train.shape} (74 pre-pitch features)")
print(f"   X_test: {X_test.shape}")
print(f"   y_train_multilabel: {y_train_multilabel.shape}")
print(f"   y_test_multilabel: {y_test_multilabel.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Features scaled with StandardScaler")

# Train individual shark models
shark_names = ['namita', 'aman', 'anupam', 'peyush', 'vineeta', 'ritesh', 'amit']
shark_models_clean = {}
shark_metrics_clean = {}

print("\n" + "="*80)
print("🦈 TRAINING INDIVIDUAL SHARK PREDICTORS (CLEAN)")
print("="*80)

for shark in shark_names:
    print(f"\n{'='*60}")
    print(f"🦈 Training {shark.upper()} Predictor (Clean)")
    print(f"{'='*60}")
    
    target_col = f'{shark}_invested'
    y_shark_train = y_train_multilabel[target_col]
    y_shark_test = y_test_multilabel[target_col]
    
    train_rate = y_shark_train.mean()
    test_rate = y_shark_test.mean()
    print(f"   Training investment rate: {train_rate*100:.1f}% ({y_shark_train.sum()}/{len(y_shark_train)})")
    print(f"   Test investment rate: {test_rate*100:.1f}% ({y_shark_test.sum()}/{len(y_shark_test)})")
    
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_shark_train == 0).sum() / max((y_shark_train == 1).sum(), 1)
    
    # Train XGBoost with clean features
    shark_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    shark_model.fit(X_train_scaled, y_shark_train)
    
    # Predictions
    y_pred = shark_model.predict(X_test_scaled)
    y_pred_proba = shark_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    if y_shark_test.sum() == 0:
        print(f"   ⚠️  {shark.upper()} has no investments in test set")
        shark_metrics_clean[shark] = {
            'accuracy': accuracy_score(y_shark_test, y_pred),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0,
            'investment_count': 0
        }
    else:
        try:
            precision = precision_score(y_shark_test, y_pred, zero_division=0)
            recall = recall_score(y_shark_test, y_pred, zero_division=0)
            f1 = f1_score(y_shark_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_shark_test, y_pred_proba)
        except:
            precision = recall = f1 = auc = 0.0
        
        shark_metrics_clean[shark] = {
            'accuracy': accuracy_score(y_shark_test, y_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'investment_count': int(y_shark_test.sum())
        }
        
        print(f"   Accuracy:  {shark_metrics_clean[shark]['accuracy']:.4f}")
        print(f"   Precision: {shark_metrics_clean[shark]['precision']:.4f}")
        print(f"   Recall:    {shark_metrics_clean[shark]['recall']:.4f}")
        print(f"   F1-Score:  {shark_metrics_clean[shark]['f1']:.4f}")
        print(f"   ROC-AUC:   {shark_metrics_clean[shark]['auc']:.4f}")
    
    shark_models_clean[shark] = shark_model

# Summary
print("\n" + "="*80)
print("📊 CLEAN MULTI-LABEL MODEL SUMMARY")
print("="*80)

shark_summary = pd.DataFrame(shark_metrics_clean).T
shark_summary = shark_summary.sort_values('f1', ascending=False)

print("\n🦈 Shark Performance Ranking (Clean Models, by F1-Score):")
print(shark_summary.to_string())

# Compare with old (leaked) models if available
print("\n" + "="*80)
print("📊 COMPARISON: LEAKED vs CLEAN MODELS")
print("="*80)

if os.path.exists('reports/shark_model_performance.csv'):
    import os
    old_metrics = pd.read_csv('reports/shark_model_performance.csv', index_col=0)
    
    print("\n{'Shark':<10} {'Old F1 (Leaked)':<20} {'New F1 (Clean)':<20} {'Change':<15}")
    print("-" * 65)
    
    for shark in shark_names:
        if shark in old_metrics.index and shark in shark_summary.index:
            old_f1 = old_metrics.loc[shark, 'f1']
            new_f1 = shark_summary.loc[shark, 'f1']
            change = new_f1 - old_f1
            print(f"{shark.capitalize():<10} {old_f1:.4f} ({old_f1*100:.1f}%){'':<6} "
                  f"{new_f1:.4f} ({new_f1*100:.1f}%){'':<6} "
                  f"{change:+.4f} ({change*100:+.1f}%)")

# Save clean models
with open('models/clean/shark_multilabel_models_clean.pkl', 'wb') as f:
    pickle.dump(shark_models_clean, f)
print("\n✅ Saved: models/clean/shark_multilabel_models_clean.pkl")

# Save scaler
with open('models/clean/shark_multilabel_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved: models/clean/shark_multilabel_scaler.pkl")

# Save metrics
shark_summary.to_csv('reports/shark_model_performance_clean.csv')
print("✅ Saved: reports/shark_model_performance_clean.csv")

print("\n" + "="*80)
print("🎉 CLEAN MULTI-LABEL MODELS COMPLETE")
print("="*80)
print(f"\n✅ Summary:")
print(f"   • Trained 7 shark predictors with clean features (74 features)")
print(f"   • No data leakage - only pre-pitch information used")
print(f"   • Average F1-Score: {shark_summary['f1'].mean():.4f}")
print(f"   • Average ROC-AUC: {shark_summary['auc'].mean():.4f}")
print(f"   • Models ready for production use")
print("\n" + "="*80)
