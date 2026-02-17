"""
COMPREHENSIVE HYPERPARAMETER TUNING
Optimize clean models for maximum performance, especially ROC-AUC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from scipy.stats import uniform, randint
import pickle
import os
import warnings
import time

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

print("="*80)
print("🎯 COMPREHENSIVE HYPERPARAMETER TUNING")
print("="*80)
print("Goal: Maximize ROC-AUC while maintaining good F1-Score")
print("="*80)

# Load clean data
X_train = pd.read_csv('data/processed/X_train_clean.csv')
X_test = pd.read_csv('data/processed/X_test_clean.csv')
y_train = pd.read_csv('data/processed/y_train_binary.csv')['got_offer']
y_test = pd.read_csv('data/processed/y_test_binary.csv')['got_offer']

print(f"\n✅ Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

# Store results
tuning_results = {}

# ============================================================================
# BASELINE PERFORMANCE (FROM PREVIOUS TRAINING)
# ============================================================================
print("\n" + "="*80)
print("📊 BASELINE PERFORMANCE (Before Tuning)")
print("="*80)

baseline = {
    'Logistic Regression': {'f1': 0.8205, 'roc_auc': 0.5593, 'accuracy': 0.7021},
    'Random Forest': {'f1': 0.7122, 'roc_auc': 0.5026, 'accuracy': 0.5816},
    'XGBoost': {'f1': 0.6735, 'roc_auc': 0.4923, 'accuracy': 0.5461},
    'LightGBM': {'f1': 0.6735, 'roc_auc': 0.4911, 'accuracy': 0.5461}
}

print("\nBaseline Metrics:")
for model, metrics in baseline.items():
    print(f"   {model:20s} - F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

# ============================================================================
# 1. LOGISTIC REGRESSION - EXTENSIVE TUNING
# ============================================================================
print("\n" + "="*80)
print("🔧 1. LOGISTIC REGRESSION - Hyperparameter Tuning")
print("="*80)

print("\n📊 Tuning Logistic Regression...")
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}],
    'max_iter': [1000, 2000]
}

lr_grid = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
lr_grid.fit(X_train, y_train)
lr_time = time.time() - start_time

print(f"\n✅ Best Parameters: {lr_grid.best_params_}")
print(f"   Best CV ROC-AUC: {lr_grid.best_score_:.4f}")
print(f"   Training time: {lr_time:.1f}s")

# Evaluate on test set
lr_tuned = lr_grid.best_estimator_
lr_pred = lr_tuned.predict(X_test)
lr_proba = lr_tuned.predict_proba(X_test)[:, 1]

tuning_results['Logistic Regression'] = {
    'model': lr_tuned,
    'params': lr_grid.best_params_,
    'cv_roc_auc': lr_grid.best_score_,
    'test_accuracy': accuracy_score(y_test, lr_pred),
    'test_precision': precision_score(y_test, lr_pred, zero_division=0),
    'test_recall': recall_score(y_test, lr_pred, zero_division=0),
    'test_f1': f1_score(y_test, lr_pred, zero_division=0),
    'test_roc_auc': roc_auc_score(y_test, lr_proba),
    'predictions': lr_proba
}

print(f"\n📊 Test Set Performance:")
print(f"   Accuracy:  {tuning_results['Logistic Regression']['test_accuracy']:.4f}")
print(f"   F1-Score:  {tuning_results['Logistic Regression']['test_f1']:.4f}")
print(f"   ROC-AUC:   {tuning_results['Logistic Regression']['test_roc_auc']:.4f}")

# ============================================================================
# 2. RANDOM FOREST - RANDOMIZED SEARCH
# ============================================================================
print("\n" + "="*80)
print("🌲 2. RANDOM FOREST - Hyperparameter Tuning")
print("="*80)

print("\n📊 Randomized Search for Random Forest...")
rf_param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'class_weight': [None, 'balanced', 'balanced_subsample'],
    'bootstrap': [True, False]
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_dist,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

start_time = time.time()
rf_random.fit(X_train, y_train)
rf_time = time.time() - start_time

print(f"\n✅ Best Parameters: {rf_random.best_params_}")
print(f"   Best CV ROC-AUC: {rf_random.best_score_:.4f}")
print(f"   Training time: {rf_time:.1f}s")

# Evaluate on test set
rf_tuned = rf_random.best_estimator_
rf_pred = rf_tuned.predict(X_test)
rf_proba = rf_tuned.predict_proba(X_test)[:, 1]

tuning_results['Random Forest'] = {
    'model': rf_tuned,
    'params': rf_random.best_params_,
    'cv_roc_auc': rf_random.best_score_,
    'test_accuracy': accuracy_score(y_test, rf_pred),
    'test_precision': precision_score(y_test, rf_pred, zero_division=0),
    'test_recall': recall_score(y_test, rf_pred, zero_division=0),
    'test_f1': f1_score(y_test, rf_pred, zero_division=0),
    'test_roc_auc': roc_auc_score(y_test, rf_proba),
    'predictions': rf_proba
}

print(f"\n📊 Test Set Performance:")
print(f"   Accuracy:  {tuning_results['Random Forest']['test_accuracy']:.4f}")
print(f"   F1-Score:  {tuning_results['Random Forest']['test_f1']:.4f}")
print(f"   ROC-AUC:   {tuning_results['Random Forest']['test_roc_auc']:.4f}")

# ============================================================================
# 3. XGBOOST - EXTENSIVE TUNING
# ============================================================================
print("\n" + "="*80)
print("🚀 3. XGBOOST - Hyperparameter Tuning")
print("="*80)

print("\n📊 Randomized Search for XGBoost...")
xgb_param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3],
    'scale_pos_weight': [1, 2, 3, 4]
}

xgb_random = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    xgb_param_dist,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

start_time = time.time()
xgb_random.fit(X_train, y_train)
xgb_time = time.time() - start_time

print(f"\n✅ Best Parameters: {xgb_random.best_params_}")
print(f"   Best CV ROC-AUC: {xgb_random.best_score_:.4f}")
print(f"   Training time: {xgb_time:.1f}s")

# Evaluate on test set
xgb_tuned = xgb_random.best_estimator_
xgb_pred = xgb_tuned.predict(X_test)
xgb_proba = xgb_tuned.predict_proba(X_test)[:, 1]

tuning_results['XGBoost'] = {
    'model': xgb_tuned,
    'params': xgb_random.best_params_,
    'cv_roc_auc': xgb_random.best_score_,
    'test_accuracy': accuracy_score(y_test, xgb_pred),
    'test_precision': precision_score(y_test, xgb_pred, zero_division=0),
    'test_recall': recall_score(y_test, xgb_pred, zero_division=0),
    'test_f1': f1_score(y_test, xgb_pred, zero_division=0),
    'test_roc_auc': roc_auc_score(y_test, xgb_proba),
    'predictions': xgb_proba
}

print(f"\n📊 Test Set Performance:")
print(f"   Accuracy:  {tuning_results['XGBoost']['test_accuracy']:.4f}")
print(f"   F1-Score:  {tuning_results['XGBoost']['test_f1']:.4f}")
print(f"   ROC-AUC:   {tuning_results['XGBoost']['test_roc_auc']:.4f}")

# ============================================================================
# 4. LIGHTGBM - EXTENSIVE TUNING
# ============================================================================
print("\n" + "="*80)
print("💡 4. LIGHTGBM - Hyperparameter Tuning")
print("="*80)

print("\n📊 Randomized Search for LightGBM...")
lgbm_param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, -1],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_samples': [5, 10, 20, 30],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0, 0.1, 0.5, 1.0],
    'class_weight': [None, 'balanced']
}

lgbm_random = RandomizedSearchCV(
    LGBMClassifier(random_state=42, verbose=-1),
    lgbm_param_dist,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

start_time = time.time()
lgbm_random.fit(X_train, y_train)
lgbm_time = time.time() - start_time

print(f"\n✅ Best Parameters: {lgbm_random.best_params_}")
print(f"   Best CV ROC-AUC: {lgbm_random.best_score_:.4f}")
print(f"   Training time: {lgbm_time:.1f}s")

# Evaluate on test set
lgbm_tuned = lgbm_random.best_estimator_
lgbm_pred = lgbm_tuned.predict(X_test)
lgbm_proba = lgbm_tuned.predict_proba(X_test)[:, 1]

tuning_results['LightGBM'] = {
    'model': lgbm_tuned,
    'params': lgbm_random.best_params_,
    'cv_roc_auc': lgbm_random.best_score_,
    'test_accuracy': accuracy_score(y_test, lgbm_pred),
    'test_precision': precision_score(y_test, lgbm_pred, zero_division=0),
    'test_recall': recall_score(y_test, lgbm_pred, zero_division=0),
    'test_f1': f1_score(y_test, lgbm_pred, zero_division=0),
    'test_roc_auc': roc_auc_score(y_test, lgbm_proba),
    'predictions': lgbm_proba
}

print(f"\n📊 Test Set Performance:")
print(f"   Accuracy:  {tuning_results['LightGBM']['test_accuracy']:.4f}")
print(f"   F1-Score:  {tuning_results['LightGBM']['test_f1']:.4f}")
print(f"   ROC-AUC:   {tuning_results['LightGBM']['test_roc_auc']:.4f}")

# ============================================================================
# 5. GRADIENT BOOSTING - TUNING
# ============================================================================
print("\n" + "="*80)
print("📈 5. GRADIENT BOOSTING - Hyperparameter Tuning")
print("="*80)

print("\n📊 Randomized Search for Gradient Boosting...")
gb_param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'subsample': [0.6, 0.8, 1.0],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5]
}

gb_random = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_dist,
    n_iter=30,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

start_time = time.time()
gb_random.fit(X_train, y_train)
gb_time = time.time() - start_time

print(f"\n✅ Best Parameters: {gb_random.best_params_}")
print(f"   Best CV ROC-AUC: {gb_random.best_score_:.4f}")
print(f"   Training time: {gb_time:.1f}s")

# Evaluate on test set
gb_tuned = gb_random.best_estimator_
gb_pred = gb_tuned.predict(X_test)
gb_proba = gb_tuned.predict_proba(X_test)[:, 1]

tuning_results['Gradient Boosting'] = {
    'model': gb_tuned,
    'params': gb_random.best_params_,
    'cv_roc_auc': gb_random.best_score_,
    'test_accuracy': accuracy_score(y_test, gb_pred),
    'test_precision': precision_score(y_test, gb_pred, zero_division=0),
    'test_recall': recall_score(y_test, gb_pred, zero_division=0),
    'test_f1': f1_score(y_test, gb_pred, zero_division=0),
    'test_roc_auc': roc_auc_score(y_test, gb_proba),
    'predictions': gb_proba
}

print(f"\n📊 Test Set Performance:")
print(f"   Accuracy:  {tuning_results['Gradient Boosting']['test_accuracy']:.4f}")
print(f"   F1-Score:  {tuning_results['Gradient Boosting']['test_f1']:.4f}")
print(f"   ROC-AUC:   {tuning_results['Gradient Boosting']['test_roc_auc']:.4f}")

print("\n" + "="*80)
print("Continuing to ensemble methods...")
print("="*80)
