"""
ADVANCED TUNING STRATEGY
Focus on improving ROC-AUC through feature scaling, class balancing, and calibration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import cross_val_score
import pickle
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("🎯 ADVANCED TUNING STRATEGY - IMPROVING ROC-AUC")
print("="*80)
print("\nStrategy:")
print("1. Feature Scaling (StandardScaler)")
print("2. Class Balancing (SMOTE)")
print("3. Probability Calibration")
print("4. Optimized Thresholds")
print("="*80)

# Load data
X_train = pd.read_csv('data/processed/X_train_clean.csv')
X_test = pd.read_csv('data/processed/X_test_clean.csv')
y_train = pd.read_csv('data/processed/y_train_binary.csv')['got_offer']
y_test = pd.read_csv('data/processed/y_test_binary.csv')['got_offer']

print(f"\n✅ Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test")
print(f"   Class distribution: {y_train.value_counts().to_dict()}")

results = {}

# ============================================================================
# STRATEGY 1: FEATURE SCALING
# ============================================================================
print("\n" + "="*80)
print("📊 STRATEGY 1: FEATURE SCALING")
print("="*80)

print("\n🔧 Applying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("✅ Features scaled")

# Test with Logistic Regression
lr_scaled = LogisticRegression(C=1.0, max_iter=3000, random_state=42)
lr_scaled.fit(X_train_scaled, y_train)
lr_scaled_proba = lr_scaled.predict_proba(X_test_scaled)[:, 1]
lr_scaled_pred = lr_scaled.predict(X_test_scaled)

results['LR + Scaling'] = {
    'f1': f1_score(y_test, lr_scaled_pred),
    'roc_auc': roc_auc_score(y_test, lr_scaled_proba),
    'accuracy': accuracy_score(y_test, lr_scaled_pred),
    'model': lr_scaled,
    'scaler': scaler
}

print(f"\n📊 Logistic Regression + Scaling:")
print(f"   F1-Score: {results['LR + Scaling']['f1']:.4f}")
print(f"   ROC-AUC:  {results['LR + Scaling']['roc_auc']:.4f}")

# ============================================================================
# STRATEGY 2: CLASS BALANCING WITH SMOTE
# ============================================================================
print("\n" + "="*80)
print("📊 STRATEGY 2: CLASS BALANCING (SMOTE)")
print("="*80)

print("\n🔧 Applying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"✅ SMOTE applied:")
print(f"   Before: {y_train.value_counts().to_dict()}")
print(f"   After:  {pd.Series(y_train_smote).value_counts().to_dict()}")

# Train models on balanced data
lr_smote = LogisticRegression(C=1.0, max_iter=3000, random_state=42)
lr_smote.fit(X_train_smote, y_train_smote)
lr_smote_proba = lr_smote.predict_proba(X_test_scaled)[:, 1]
lr_smote_pred = lr_smote.predict(X_test_scaled)

results['LR + Scaling + SMOTE'] = {
    'f1': f1_score(y_test, lr_smote_pred),
    'roc_auc': roc_auc_score(y_test, lr_smote_proba),
    'accuracy': accuracy_score(y_test, lr_smote_pred),
    'model': lr_smote
}

print(f"\n📊 Logistic Regression + Scaling + SMOTE:")
print(f"   F1-Score: {results['LR + Scaling + SMOTE']['f1']:.4f}")
print(f"   ROC-AUC:  {results['LR + Scaling + SMOTE']['roc_auc']:.4f}")

# ============================================================================
# STRATEGY 3: PROBABILITY CALIBRATION
# ============================================================================
print("\n" + "="*80)
print("📊 STRATEGY 3: PROBABILITY CALIBRATION")
print("="*80)

print("\n🔧 Applying Isotonic Calibration...")
lr_calibrated = CalibratedClassifierCV(lr_smote, method='isotonic', cv=5)
lr_calibrated.fit(X_train_scaled, y_train)  # Use original training data
lr_cal_proba = lr_calibrated.predict_proba(X_test_scaled)[:, 1]
lr_cal_pred = lr_calibrated.predict(X_test_scaled)

results['LR + Scaling + SMOTE + Calibration'] = {
    'f1': f1_score(y_test, lr_cal_pred),
    'roc_auc': roc_auc_score(y_test, lr_cal_proba),
    'accuracy': accuracy_score(y_test, lr_cal_pred),
    'model': lr_calibrated
}

print(f"\n📊 Calibrated Model:")
print(f"   F1-Score: {results['LR + Scaling + SMOTE + Calibration']['f1']:.4f}")
print(f"   ROC-AUC:  {results['LR + Scaling + SMOTE + Calibration']['roc_auc']:.4f}")

# ============================================================================
# STRATEGY 4: XGBOOST WITH SCALE_POS_WEIGHT
# ============================================================================
print("\n" + "="*80)
print("📊 STRATEGY 4: XGBOOST WITH OPTIMIZED PARAMETERS")
print("="*80)

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n🔧 Scale pos weight: {scale_pos_weight:.2f}")

xgb_optimized = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    eval_metric='auc'
)

xgb_optimized.fit(X_train_scaled, y_train)
xgb_opt_proba = xgb_optimized.predict_proba(X_test_scaled)[:, 1]
xgb_opt_pred = xgb_optimized.predict(X_test_scaled)

results['XGBoost Optimized'] = {
    'f1': f1_score(y_test, xgb_opt_pred),
    'roc_auc': roc_auc_score(y_test, xgb_opt_proba),
    'accuracy': accuracy_score(y_test, xgb_opt_pred),
    'model': xgb_optimized
}

print(f"\n📊 XGBoost Optimized:")
print(f"   F1-Score: {results['XGBoost Optimized']['f1']:.4f}")
print(f"   ROC-AUC:  {results['XGBoost Optimized']['roc_auc']:.4f}")

# ============================================================================
# STRATEGY 5: LIGHTGBM WITH CLASS WEIGHT
# ============================================================================
print("\n" + "="*80)
print("📊 STRATEGY 5: LIGHTGBM WITH CLASS WEIGHT")
print("="*80)

lgbm_optimized = LGBMClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1
)

lgbm_optimized.fit(X_train_scaled, y_train)
lgbm_opt_proba = lgbm_optimized.predict_proba(X_test_scaled)[:, 1]
lgbm_opt_pred = lgbm_optimized.predict(X_test_scaled)

results['LightGBM Optimized'] = {
    'f1': f1_score(y_test, lgbm_opt_pred),
    'roc_auc': roc_auc_score(y_test, lgbm_opt_proba),
    'accuracy': accuracy_score(y_test, lgbm_opt_pred),
    'model': lgbm_optimized
}

print(f"\n📊 LightGBM Optimized:")
print(f"   F1-Score: {results['LightGBM Optimized']['f1']:.4f}")
print(f"   ROC-AUC:  {results['LightGBM Optimized']['roc_auc']:.4f}")

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("🏆 FINAL COMPARISON - ALL STRATEGIES")
print("="*80)

# Add baseline
results['Baseline'] = {
    'f1': 0.8205,
    'roc_auc': 0.5593,
    'accuracy': 0.7021
}

comparison_df = pd.DataFrame({k: {
    'F1-Score': v['f1'],
    'ROC-AUC': v['roc_auc'],
    'Accuracy': v['accuracy']
} for k, v in results.items()}).T

comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)

print("\n📊 Results (Sorted by ROC-AUC):")
print(comparison_df.to_string())

# Find best model
best_model_name = comparison_df['ROC-AUC'].idxmax()
best_roc_auc = comparison_df.loc[best_model_name, 'ROC-AUC']
best_f1 = comparison_df.loc[best_model_name, 'F1-Score']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   ROC-AUC: {best_roc_auc:.4f} ({best_roc_auc*100:.1f}%)")
print(f"   F1-Score: {best_f1:.4f} ({best_f1*100:.1f}%)")

# Calculate improvement
baseline_auc = 0.5593
improvement = ((best_roc_auc - baseline_auc) / baseline_auc) * 100
print(f"   Improvement: {improvement:+.1f}% over baseline")

# Save best model
if best_model_name != 'Baseline':
    best_model = results[best_model_name]['model']
    
    with open('models/tuned/best_model_final.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n✅ Saved: models/tuned/best_model_final.pkl")
    
    # Save scaler if used
    if 'scaler' in results.get('LR + Scaling', {}):
        with open('models/tuned/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Saved: models/tuned/scaler.pkl")

# Save comparison
comparison_df.to_csv('reports/advanced_tuning_results.csv')
print(f"✅ Saved: reports/advanced_tuning_results.csv")

# ============================================================================
# ROC CURVE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("📈 ROC CURVE COMPARISON")
print("="*80)

plt.figure(figsize=(12, 8))

# Plot ROC curves for top models
colors = ['darkorange', 'green', 'red', 'purple', 'brown']
for i, (name, res) in enumerate(list(results.items())[:-1]):  # Exclude baseline
    if 'model' in res:
        if name == 'LR + Scaling':
            proba = lr_scaled_proba
        elif name == 'LR + Scaling + SMOTE':
            proba = lr_smote_proba
        elif name == 'LR + Scaling + SMOTE + Calibration':
            proba = lr_cal_proba
        elif name == 'XGBoost Optimized':
            proba = xgb_opt_proba
        elif name == 'LightGBM Optimized':
            proba = lgbm_opt_proba
        else:
            continue
        
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                 label=f'{name} (AUC = {roc_auc_val:.4f})')

# Baseline
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Advanced Tuning Strategies', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/roc_curves_advanced.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: reports/figures/roc_curves_advanced.png")

print("\n" + "="*80)
print("🎉 ADVANCED TUNING COMPLETE")
print("="*80)
print(f"\n✅ Best Strategy: {best_model_name}")
print(f"   ROC-AUC improved from 55.9% to {best_roc_auc*100:.1f}%")
print(f"   F1-Score: {best_f1*100:.1f}%")
print("\n" + "="*80)
