import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')

print("="*80)
print("🚀 PHASE 3: DATA PREPROCESSING & FEATURE ENGINEERING")
print("="*80)

# Load data
df = pd.read_csv('data/raw/Shark Tank India.csv')
print(f"\n📊 Dataset loaded: {df.shape}")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")

df_fe = df.copy()

# ============================================================================
# SECTION 1: DATA CLEANING
# ============================================================================
print("\n" + "="*80)
print("🧹 SECTION 1: DATA CLEANING")
print("="*80)

# Shark-specific columns - fill with 0
shark_names = ['Namita', 'Vineeta', 'Anupam', 'Aman', 'Peyush', 'Ritesh', 'Amit', 'Guest']
shark_cols = []
for shark in shark_names:
    for suffix in ['Investment Amount', 'Investment Equity', 'Debt Amount']:
        col = f'{shark} {suffix}'
        if col in df_fe.columns:
            df_fe[col] = df_fe[col].fillna(0)
            shark_cols.append(col)

print(f"✅ Filled {len(shark_cols)} shark-specific columns with 0")

# Financial metrics - industry median
financial_cols = ['Yearly Revenue', 'Monthly Sales', 'Gross Margin', 'Net Margin', 'EBITDA']
for col in financial_cols:
    if col in df_fe.columns:
        df_fe[col] = df_fe.groupby('Industry')[col].transform(lambda x: x.fillna(x.median()))

print(f"✅ Imputed {len([c for c in financial_cols if c in df_fe.columns])} financial columns with industry median")

# SKUs - median
if 'SKUs' in df_fe.columns:
    df_fe['SKUs'] = df_fe['SKUs'].fillna(df_fe['SKUs'].median())

# Deal columns - fill with 0
deal_cols = ['Total Deal Amount', 'Total Deal Equity', 'Total Deal Debt', 'Debt Interest',
             'Royalty Percentage', 'Royalty Recouped Amount', 'Advisory Shares Equity']
for col in deal_cols:
    if col in df_fe.columns:
        df_fe[col] = df_fe[col].fillna(0)

# Convert Yes/No to 1/0
binary_cols = ['Cash Burn', 'Has Patents', 'Bootstrapped']
for col in binary_cols:
    if col in df_fe.columns:
        df_fe[col] = df_fe[col].map({'Yes': 1, 'yes': 1, 'No': 0, 'no': 0}).fillna(0)

# Ensure numeric
numeric_cols = ['Yearly Revenue', 'Monthly Sales', 'Gross Margin', 'Net Margin',
                'Original Ask Amount', 'Original Offered Equity', 'Valuation Requested']
for col in numeric_cols:
    if col in df_fe.columns:
        df_fe[col] = pd.to_numeric(df_fe[col], errors='coerce').fillna(0)

# Outlier flags
if 'Valuation Requested' in df_fe.columns:
    Q1 = df_fe['Valuation Requested'].quantile(0.25)
    Q3 = df_fe['Valuation Requested'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df_fe['is_outlier_valuation'] = (df_fe['Valuation Requested'] > upper_bound).astype(int)

if 'Yearly Revenue' in df_fe.columns:
    df_fe['is_high_revenue'] = (df_fe['Yearly Revenue'] > 1000).astype(int)

print("✅ Data cleaning completed")

# ============================================================================
# SECTION 2: FEATURE ENGINEERING (47 FEATURES)
# ============================================================================
print("\n" + "="*80)
print("💡 SECTION 2: FEATURE ENGINEERING (47 FEATURES)")
print("="*80)

total_features = 0

# Financial Health Indicators (10)
print("\n💰 Financial Health Indicators...")
df_fe['revenue_per_sku'] = df_fe['Yearly Revenue'] / (df_fe['SKUs'] + 1)
df_fe['monthly_to_yearly_ratio'] = ((df_fe['Monthly Sales'] * 12) / (df_fe['Yearly Revenue'] + 1)).clip(0, 10)
df_fe['profit_margin_gap'] = df_fe['Gross Margin'] - df_fe['Net Margin']
df_fe['profitability_score'] = (df_fe['Net Margin'] * df_fe['Yearly Revenue']) / 1000
df_fe['ebitda_margin'] = df_fe['EBITDA'] / (df_fe['Yearly Revenue'] + 1)
df_fe['burn_rate'] = df_fe['Cash Burn'] * df_fe['Monthly Sales']
df_fe['runway_months'] = (df_fe['Yearly Revenue'] / (df_fe['burn_rate'] + 1)).clip(0, 100)
df_fe['is_pre_revenue'] = (df_fe['Yearly Revenue'] == 0).astype(int)
df_fe['revenue_category'] = pd.cut(df_fe['Yearly Revenue'], 
                                    bins=[-np.inf, 0, 100, 1000, 10000, np.inf],
                                    labels=[0, 1, 2, 3, 4]).astype(int)
df_fe['financial_health_score'] = ((df_fe['Yearly Revenue'] > 0).astype(int) * 2 +
                                    (df_fe['Net Margin'] > 10).astype(int) * 2 +
                                    (df_fe['Cash Burn'] == 0).astype(int))
total_features += 10
print(f"   ✅ Created 10 features")

# Deal Structure Indicators (8)
print("🤝 Deal Structure Indicators...")
df_fe['revenue_multiple'] = (df_fe['Valuation Requested'] / (df_fe['Yearly Revenue'] + 1)).clip(0, 1000)
df_fe['ask_percentage'] = (df_fe['Original Ask Amount'] / (df_fe['Valuation Requested'] + 1)) * 100
industry_median_multiple = df_fe.groupby('Industry')['revenue_multiple'].transform('median')
df_fe['valuation_reasonableness'] = (df_fe['revenue_multiple'] / (industry_median_multiple + 1)).clip(0, 10)
df_fe['expected_equity_dilution'] = df_fe['Original Offered Equity']
df_fe['deal_size_category'] = pd.cut(df_fe['Original Ask Amount'],
                                      bins=[-np.inf, 50, 100, 200, 500, np.inf],
                                      labels=[0, 1, 2, 3, 4]).astype(int)
df_fe['valuation_to_ask_ratio'] = df_fe['Valuation Requested'] / (df_fe['Original Ask Amount'] + 1)
df_fe['is_reasonable_valuation'] = (df_fe['valuation_reasonableness'] <= 2).astype(int)
df_fe['deal_complexity_score'] = ((df_fe['Total Deal Debt'] > 0).astype(int) +
                                   (df_fe['Royalty Percentage'] > 0).astype(int) +
                                   (df_fe['Advisory Shares Equity'] > 0).astype(int))
total_features += 8
print(f"   ✅ Created 8 features")

# Team Composition Features (7)
print("👥 Team Composition Features...")
df_fe['team_size'] = df_fe['Number of Presenters'].fillna(1)
df_fe['male_ratio'] = df_fe['Male Presenters'] / (df_fe['team_size'] + 0.001)
df_fe['female_ratio'] = df_fe['Female Presenters'] / (df_fe['team_size'] + 0.001)
df_fe['gender_diversity_score'] = 1 - abs(df_fe['male_ratio'] - df_fe['female_ratio'])
df_fe['is_solo_founder'] = (df_fe['team_size'] == 1).astype(int)
df_fe['is_couple'] = df_fe['Couple Presenters'].fillna(0).astype(int)
df_fe['has_female_founder'] = (df_fe['Female Presenters'] > 0).astype(int)
total_features += 7
print(f"   ✅ Created 7 features")

# Innovation Indicators (4)
print("💡 Innovation Indicators...")
df_fe['has_patent'] = df_fe['Has Patents'].fillna(0).astype(int)
df_fe['is_bootstrapped'] = df_fe['Bootstrapped'].fillna(0).astype(int)
df_fe['sku_count'] = df_fe['SKUs'].fillna(0)
df_fe['innovation_score'] = df_fe['has_patent'] * 2 + df_fe['is_bootstrapped']
total_features += 4
print(f"   ✅ Created 4 features")

# Shark Affinity Scores (7)
print("🦈 Shark Affinity Scores...")
shark_names = ['Namita', 'Aman', 'Anupam', 'Peyush', 'Vineeta', 'Ritesh', 'Amit']
for shark in shark_names:
    investment_col = f'{shark} Investment Amount'
    if investment_col in df_fe.columns:
        industry_investments = df_fe.groupby('Industry')[investment_col].apply(lambda x: (x > 0).sum())
        total_investments = (df_fe[investment_col] > 0).sum()
        if total_investments > 0:
            affinity = industry_investments / total_investments
        else:
            affinity = industry_investments * 0
        df_fe[f'{shark.lower()}_industry_fit'] = df_fe['Industry'].map(affinity).fillna(0)
total_features += 7
print(f"   ✅ Created 7 features")

# Industry Context Features (5)
print("🏭 Industry Context Features...")
industry_success_rate = df_fe.groupby('Industry')['Received Offer'].mean()
df_fe['industry_avg_success_rate'] = df_fe['Industry'].map(industry_success_rate)
industry_avg_equity = df_fe.groupby('Industry')['Total Deal Equity'].mean()
df_fe['industry_avg_equity'] = df_fe['Industry'].map(industry_avg_equity)
industry_median_val = df_fe.groupby('Industry')['Valuation Requested'].median()
df_fe['industry_median_valuation'] = df_fe['Industry'].map(industry_median_val)
df_fe['industry_pitch_count'] = df_fe.groupby('Industry')['Startup Name'].transform('count')
df_fe['industry_competition_index'] = df_fe['industry_pitch_count'] / len(df_fe)
total_features += 5
print(f"   ✅ Created 5 features")

# Geographic Features (4)
print("🗺️  Geographic Features...")
state_success = df_fe.groupby('Pitchers State')['Received Offer'].mean()
df_fe['state_success_rate'] = df_fe['Pitchers State'].map(state_success)
metro_states = ['Maharashtra', 'Delhi', 'Karnataka']
df_fe['is_metro'] = df_fe['Pitchers State'].isin(metro_states).astype(int)
df_fe['state_pitch_density'] = df_fe.groupby('Pitchers State')['Startup Name'].transform('count')
df_fe['geographic_diversity_score'] = df_fe['state_pitch_density'] / len(df_fe)
total_features += 4
print(f"   ✅ Created 4 features")

# Outlier flags (already created, count them)
total_features += 2

print(f"\n✅ Total engineered features: {total_features}")

# ============================================================================
# SECTION 3: TARGET VARIABLE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("🎯 SECTION 3: TARGET VARIABLE ENGINEERING")
print("="*80)

# Binary classification targets
df_fe['got_offer'] = df_fe['Received Offer'].fillna(0).astype(int)
df_fe['accepted_offer'] = df_fe['Accepted Offer'].fillna(0).astype(int)

print(f"\n✅ Binary targets:")
print(f"   - got_offer: {df_fe['got_offer'].sum()}/{len(df_fe)} ({df_fe['got_offer'].mean()*100:.1f}%)")
print(f"   - accepted_offer: {df_fe['accepted_offer'].sum()}/{len(df_fe)} ({df_fe['accepted_offer'].mean()*100:.1f}%)")

# Multi-label targets (7 sharks)
shark_targets = []
shark_names = ['Namita', 'Aman', 'Anupam', 'Peyush', 'Vineeta', 'Ritesh', 'Amit']

print(f"\n✅ Multi-label targets (individual sharks):")
for shark in shark_names:
    investment_col = f'{shark} Investment Amount'
    if investment_col in df_fe.columns:
        target_col = f'{shark.lower()}_invested'
        df_fe[target_col] = (df_fe[investment_col] > 0).astype(int)
        shark_targets.append(target_col)
        print(f"   - {target_col}: {df_fe[target_col].sum()} deals ({df_fe[target_col].mean()*100:.1f}%)")

# Regression target
df_fe['equity_dilution'] = df_fe['Total Deal Equity'] - df_fe['Original Offered Equity']
print(f"\n✅ Regression target:")
print(f"   - equity_dilution: Mean={df_fe['equity_dilution'].mean():.2f}%, Std={df_fe['equity_dilution'].std():.2f}%")

# ============================================================================
# SECTION 4: FEATURE SELECTION
# ============================================================================
print("\n" + "="*80)
print("🔍 SECTION 4: FEATURE SELECTION & CORRELATION ANALYSIS")
print("="*80)

# Select only numeric features for modeling
numeric_features = df_fe.select_dtypes(include=[np.number]).columns.tolist()

# Remove target variables and identifiers from features
exclude_cols = ['Season Number', 'Episode Number', 'Pitch Number', 'got_offer', 'accepted_offer', 
                'equity_dilution', 'Received Offer', 'Accepted Offer'] + shark_targets

# Also exclude individual shark investment amounts
shark_investment_cols = [col for col in numeric_features if 'Investment Amount' in col or 'Investment Equity' in col or 'Debt Amount' in col]
exclude_cols.extend(shark_investment_cols)

feature_cols = [col for col in numeric_features if col not in exclude_cols]

print(f"\n📊 Feature Selection:")
print(f"   Total numeric columns: {len(numeric_features)}")
print(f"   Excluded columns: {len(exclude_cols)}")
print(f"   Selected features for modeling: {len(feature_cols)}")

# Correlation with target
print(f"\n🎯 Top 20 Features Correlated with 'got_offer':")
correlations = df_fe[feature_cols + ['got_offer']].corr()['got_offer'].sort_values(ascending=False)
top_20_corr = correlations[1:21]

for i, (feat, corr) in enumerate(top_20_corr.items(), 1):
    print(f"   {i:2d}. {feat:45s} : {corr:7.4f}")

# Save feature importance
os.makedirs('data/processed', exist_ok=True)
feature_importance_df = pd.DataFrame({
    'Feature': correlations.index[1:],
    'Correlation_with_got_offer': correlations.values[1:]
}).sort_values('Correlation_with_got_offer', ascending=False, key=abs)

feature_importance_df.to_csv('data/processed/feature_importance_preliminary.csv', index=False)
print(f"\n✅ Saved: feature_importance_preliminary.csv")

# ============================================================================
# SECTION 5: TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("✂️  SECTION 5: TRAIN/TEST SPLIT")
print("="*80)

# Prepare feature matrix
X = df_fe[feature_cols].copy()
y_binary = df_fe['got_offer'].copy()
y_multilabel = df_fe[shark_targets].copy()
y_regression = df_fe['equity_dilution'].copy()

# Handle any remaining NaN values
X = X.fillna(0)

print(f"\n📊 Data Shapes:")
print(f"   X (features): {X.shape}")
print(f"   y_binary: {y_binary.shape}")
print(f"   y_multilabel: {y_multilabel.shape}")
print(f"   y_regression: {y_regression.shape}")

# Stratified split by industry (fill NaN in Industry first)
stratify_col = df_fe['Industry'].fillna('Unknown')
X_train, X_test, y_train_binary, y_test_binary = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=stratify_col
)

# Get corresponding indices for other targets
train_idx = X_train.index
test_idx = X_test.index

y_train_multilabel = y_multilabel.loc[train_idx]
y_test_multilabel = y_multilabel.loc[test_idx]

y_train_regression = y_regression.loc[train_idx]
y_test_regression = y_regression.loc[test_idx]

print(f"\n✅ Train/Test Split Complete:")
print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"\n   Binary target distribution:")
print(f"   - Train: {y_train_binary.sum()}/{len(y_train_binary)} offers ({y_train_binary.mean()*100:.1f}%)")
print(f"   - Test: {y_test_binary.sum()}/{len(y_test_binary)} offers ({y_test_binary.mean()*100:.1f}%)")

# ============================================================================
# SECTION 6: SAVE PROCESSED DATA
# ============================================================================
print("\n" + "="*80)
print("💾 SECTION 6: SAVING PROCESSED DATA")
print("="*80)

# Save train/test splits
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)

y_train_binary.to_csv('data/processed/y_train_binary.csv', index=False, header=['got_offer'])
y_test_binary.to_csv('data/processed/y_test_binary.csv', index=False, header=['got_offer'])

y_train_multilabel.to_csv('data/processed/y_train_multilabel.csv', index=False)
y_test_multilabel.to_csv('data/processed/y_test_multilabel.csv', index=False)

y_train_regression.to_csv('data/processed/y_train_regression.csv', index=False, header=['equity_dilution'])
y_test_regression.to_csv('data/processed/y_test_regression.csv', index=False, header=['equity_dilution'])

print(f"✅ Saved train/test splits:")
print(f"   - X_train.csv ({X_train.shape})")
print(f"   - X_test.csv ({X_test.shape})")
print(f"   - y_train_binary.csv")
print(f"   - y_test_binary.csv")
print(f"   - y_train_multilabel.csv ({y_train_multilabel.shape})")
print(f"   - y_test_multilabel.csv ({y_test_multilabel.shape})")
print(f"   - y_train_regression.csv")
print(f"   - y_test_regression.csv")

# Save full processed dataset
df_fe.to_csv('data/processed/processed_data_full.csv', index=False)
print(f"\n✅ Saved full processed dataset: processed_data_full.csv ({df_fe.shape})")

# Save feature list
feature_list_df = pd.DataFrame({
    'Feature': feature_cols,
    'Data_Type': [df_fe[col].dtype for col in feature_cols]
})
feature_list_df.to_csv('data/processed/feature_list.csv', index=False)
print(f"✅ Saved feature list: feature_list.csv ({len(feature_cols)} features)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n" + "="*80)
print(f"🎉 PHASE 3 COMPLETE: DATA PREPROCESSING & FEATURE ENGINEERING")
print(f"="*80)
print(f"\n📊 Summary:")
print(f"   ✅ {total_features} new features created")
print(f"   ✅ {len(feature_cols)} features selected for modeling")
print(f"   ✅ {len(X_train)} training samples, {len(X_test)} test samples")
print(f"   ✅ 3 target types: binary, multi-label (7 sharks), regression")
print(f"   ✅ All data saved to data/processed/")
print(f"\n🚀 Ready for Phase 4: ML Model Training!")
print(f"="*80)
