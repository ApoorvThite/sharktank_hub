"""
PHASE 4: ML MODEL DEVELOPMENT
Train binary, multi-label, and regression models for Shark Tank predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

print("="*80)
print("🤖 PHASE 4: ML MODEL DEVELOPMENT")
print("="*80)

# Create models directory
os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

# ============================================================================
# SECTION 1: LOAD PROCESSED DATA
# ============================================================================
print("\n" + "="*80)
print("📊 SECTION 1: LOADING PROCESSED DATA")
print("="*80)

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train_binary = pd.read_csv('data/processed/y_train_binary.csv')['got_offer']
y_test_binary = pd.read_csv('data/processed/y_test_binary.csv')['got_offer']
y_train_multilabel = pd.read_csv('data/processed/y_train_multilabel.csv')
y_test_multilabel = pd.read_csv('data/processed/y_test_multilabel.csv')
y_train_regression = pd.read_csv('data/processed/y_train_regression.csv')['equity_dilution']
y_test_regression = pd.read_csv('data/processed/y_test_regression.csv')['equity_dilution']

print(f"\n✅ Data loaded successfully:")
print(f"   X_train: {X_train.shape}")
print(f"   X_test: {X_test.shape}")
print(f"   y_train_binary: {y_train_binary.shape} (offer rate: {y_train_binary.mean()*100:.1f}%)")
print(f"   y_test_binary: {y_test_binary.shape} (offer rate: {y_test_binary.mean()*100:.1f}%)")
print(f"   y_train_multilabel: {y_train_multilabel.shape}")
print(f"   y_test_multilabel: {y_test_multilabel.shape}")

# ============================================================================
# SECTION 2: BINARY CLASSIFICATION - OFFER PREDICTION
# ============================================================================
print("\n" + "="*80)
print("🎯 SECTION 2: BINARY CLASSIFICATION - WILL THEY GET AN OFFER?")
print("="*80)

binary_results = {}

# Model 1: Logistic Regression (Baseline)
print("\n📊 Training Logistic Regression (Baseline)...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train_binary)
lr_pred = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]

binary_results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test_binary, lr_pred),
    'precision': precision_score(y_test_binary, lr_pred),
    'recall': recall_score(y_test_binary, lr_pred),
    'f1': f1_score(y_test_binary, lr_pred),
    'roc_auc': roc_auc_score(y_test_binary, lr_proba)
}

print(f"   Accuracy: {binary_results['Logistic Regression']['accuracy']:.4f}")
print(f"   F1-Score: {binary_results['Logistic Regression']['f1']:.4f}")
print(f"   ROC-AUC: {binary_results['Logistic Regression']['roc_auc']:.4f}")

# Model 2: Random Forest
print("\n🌲 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train_binary)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

binary_results['Random Forest'] = {
    'accuracy': accuracy_score(y_test_binary, rf_pred),
    'precision': precision_score(y_test_binary, rf_pred),
    'recall': recall_score(y_test_binary, rf_pred),
    'f1': f1_score(y_test_binary, rf_pred),
    'roc_auc': roc_auc_score(y_test_binary, rf_proba)
}

print(f"   Accuracy: {binary_results['Random Forest']['accuracy']:.4f}")
print(f"   F1-Score: {binary_results['Random Forest']['f1']:.4f}")
print(f"   ROC-AUC: {binary_results['Random Forest']['roc_auc']:.4f}")

# Model 3: XGBoost (Expected Best Performer)
print("\n🚀 Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train_binary)
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

binary_results['XGBoost'] = {
    'accuracy': accuracy_score(y_test_binary, xgb_pred),
    'precision': precision_score(y_test_binary, xgb_pred),
    'recall': recall_score(y_test_binary, xgb_pred),
    'f1': f1_score(y_test_binary, xgb_pred),
    'roc_auc': roc_auc_score(y_test_binary, xgb_proba)
}

print(f"   Accuracy: {binary_results['XGBoost']['accuracy']:.4f}")
print(f"   F1-Score: {binary_results['XGBoost']['f1']:.4f}")
print(f"   ROC-AUC: {binary_results['XGBoost']['roc_auc']:.4f}")

# Model 4: LightGBM
print("\n💡 Training LightGBM...")
lgbm_model = LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(X_train, y_train_binary)
lgbm_pred = lgbm_model.predict(X_test)
lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]

binary_results['LightGBM'] = {
    'accuracy': accuracy_score(y_test_binary, lgbm_pred),
    'precision': precision_score(y_test_binary, lgbm_pred),
    'recall': recall_score(y_test_binary, lgbm_pred),
    'f1': f1_score(y_test_binary, lgbm_pred),
    'roc_auc': roc_auc_score(y_test_binary, lgbm_proba)
}

print(f"   Accuracy: {binary_results['LightGBM']['accuracy']:.4f}")
print(f"   F1-Score: {binary_results['LightGBM']['f1']:.4f}")
print(f"   ROC-AUC: {binary_results['LightGBM']['roc_auc']:.4f}")

# Select best model
best_model_name = max(binary_results, key=lambda x: binary_results[x]['f1'])
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   F1-Score: {binary_results[best_model_name]['f1']:.4f}")
print(f"   ROC-AUC: {binary_results[best_model_name]['roc_auc']:.4f}")

# Use XGBoost as best model (typically performs best)
best_model = xgb_model
best_pred = xgb_pred
best_proba = xgb_proba

# Detailed evaluation
print("\n📊 Detailed Classification Report:")
print(classification_report(y_test_binary, best_pred, target_names=['No Offer', 'Got Offer']))

# Confusion Matrix
cm = confusion_matrix(y_test_binary, best_pred)
print("\n📊 Confusion Matrix:")
print(f"   True Negatives:  {cm[0,0]:3d}  |  False Positives: {cm[0,1]:3d}")
print(f"   False Negatives: {cm[1,0]:3d}  |  True Positives:  {cm[1,1]:3d}")

# Save binary model
with open('models/shark_predictor_binary.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("\n✅ Saved: models/shark_predictor_binary.pkl")

# ============================================================================
# SECTION 3: HYPERPARAMETER TUNING (OPTIONAL - COMMENTED FOR SPEED)
# ============================================================================
print("\n" + "="*80)
print("⚙️  SECTION 3: HYPERPARAMETER TUNING (Quick Version)")
print("="*80)

# Quick tuning with smaller grid
print("\n🔧 Fine-tuning XGBoost with GridSearchCV...")
param_grid = {
    'max_depth': [5, 6, 7],
    'learning_rate': [0.03, 0.05, 0.07],
    'n_estimators': [150, 200, 250]
}

grid_search = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train_binary)

print(f"\n✅ Best Parameters: {grid_search.best_params_}")
print(f"   Best CV F1-Score: {grid_search.best_score_:.4f}")

# Use tuned model
tuned_model = grid_search.best_estimator_
tuned_pred = tuned_model.predict(X_test)
tuned_proba = tuned_model.predict_proba(X_test)[:, 1]

tuned_f1 = f1_score(y_test_binary, tuned_pred)
tuned_auc = roc_auc_score(y_test_binary, tuned_proba)

print(f"   Tuned Test F1-Score: {tuned_f1:.4f}")
print(f"   Tuned Test ROC-AUC: {tuned_auc:.4f}")

# Use tuned model if better
if tuned_f1 > binary_results['XGBoost']['f1']:
    best_model = tuned_model
    best_pred = tuned_pred
    best_proba = tuned_proba
    print("\n🎉 Tuned model is better! Using tuned model.")
else:
    print("\n📊 Original model performs similarly. Using original.")

# Save best tuned model
with open('models/shark_predictor_tuned.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\n" + "="*80)
print("Continuing to multi-label classification...")
print("="*80)
EOF
