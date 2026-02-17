"""
PHASE 8: DEAL STRUCTURE DECODER
Comprehensive analysis of alternative deal structures (debt, royalty, advisory shares)
with predictive modeling and recommendation engine
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import warnings
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("="*80)
print("📋 PHASE 8: DEAL STRUCTURE DECODER")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading processed data...")
df = pd.read_csv('data/processed/processed_data_full.csv')

print(f"✅ Loaded {len(df)} startups")
print(f"   Deals: {df['got_offer'].sum()}")

# Define sharks
sharks = ['Namita', 'Aman', 'Anupam', 'Peyush', 'Vineeta', 'Ritesh', 'Amit']

# ============================================================================
# 8.1 DEAL STRUCTURE CLASSIFICATION
# ============================================================================
print("\n" + "="*80)
print("8.1 DEAL STRUCTURE CLASSIFICATION")
print("="*80)

# Initialize deal structure column
df['deal_structure'] = 'No Deal'

# Classify funded deals
funded = df[df['Received Offer'] == 1].copy()

print(f"\n🔍 Classifying {len(funded)} funded deals...")

for idx in funded.index:
    has_debt = False
    has_royalty = False
    has_advisory = False
    
    # Check for debt
    if 'Total Deal Debt' in df.columns:
        has_debt = pd.notna(df.loc[idx, 'Total Deal Debt']) and df.loc[idx, 'Total Deal Debt'] > 0
    
    # Check for royalty
    if 'Royalty Percentage' in df.columns:
        has_royalty = pd.notna(df.loc[idx, 'Royalty Percentage']) and df.loc[idx, 'Royalty Percentage'] > 0
    
    # Check for advisory shares
    if 'Advisory Shares Equity' in df.columns:
        has_advisory = pd.notna(df.loc[idx, 'Advisory Shares Equity']) and df.loc[idx, 'Advisory Shares Equity'] > 0
    
    # Classify
    if has_debt and has_royalty:
        df.loc[idx, 'deal_structure'] = 'Debt + Royalty + Equity'
    elif has_debt:
        df.loc[idx, 'deal_structure'] = 'Debt + Equity'
    elif has_royalty:
        df.loc[idx, 'deal_structure'] = 'Royalty + Equity'
    elif has_advisory:
        df.loc[idx, 'deal_structure'] = 'Advisory Shares'
    else:
        df.loc[idx, 'deal_structure'] = 'Pure Equity'

# Distribution
print("\n📊 Deal Structure Distribution:")
structure_counts = df[df['deal_structure'] != 'No Deal']['deal_structure'].value_counts()
for structure, count in structure_counts.items():
    pct = (count / structure_counts.sum()) * 100
    print(f"   {structure:30s}: {count:3d} deals ({pct:5.1f}%)")

# Save classification
df.to_csv('data/processed/processed_data_with_deal_structures.csv', index=False)
print("\n✅ Saved: data/processed/processed_data_with_deal_structures.csv")

# Visualize distribution
fig, ax = plt.subplots(figsize=(12, 8))
structure_counts.plot(kind='bar', color='steelblue', alpha=0.8, ax=ax)
ax.set_title('Distribution of Deal Structures', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Deal Type', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Deals', fontsize=13, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(structure_counts.values):
    ax.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('reports/figures/deal_structure_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: reports/figures/deal_structure_distribution.png")
plt.close()

# ============================================================================
# 8.2 DEBT DEAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("8.2 DEBT DEAL ANALYSIS")
print("="*80)

# Analyze deals with debt component
if 'Total Deal Debt' in df.columns:
    debt_deals = df[(df['Total Deal Debt'] > 0) & pd.notna(df['Total Deal Debt'])].copy()
    
    print(f"\n💰 DEBT DEALS OVERVIEW")
    print(f"{'='*70}")
    print(f"Total Debt Deals: {len(debt_deals)}")
    total_funded = len(df[df['Received Offer'] == 1])
    print(f"Percentage of all deals: {len(debt_deals)/total_funded*100:.1f}%")
    
    if len(debt_deals) > 0:
        # Debt characteristics
        print(f"\n📊 Debt Amount Statistics:")
        print(f"   Mean: ₹{debt_deals['Total Deal Debt'].mean():.1f}L")
        print(f"   Median: ₹{debt_deals['Total Deal Debt'].median():.1f}L")
        print(f"   Range: ₹{debt_deals['Total Deal Debt'].min():.0f}L - ₹{debt_deals['Total Deal Debt'].max():.0f}L")
        
        # Interest rates
        if 'Debt Interest' in debt_deals.columns:
            print(f"\n💸 Debt Interest Rates:")
            interest_stats = debt_deals['Debt Interest'].describe()
            print(f"   Mean: {interest_stats['mean']:.1f}%")
            print(f"   Median: {debt_deals['Debt Interest'].median():.1f}%")
            print(f"   Range: {debt_deals['Debt Interest'].min():.1f}% - {debt_deals['Debt Interest'].max():.1f}%")
        
        # Debt-to-total ratio
        debt_deals['debt_to_total_ratio'] = (
            debt_deals['Total Deal Debt'] / 
            (debt_deals['Total Deal Amount'] + debt_deals['Total Deal Debt'])
        )
        print(f"\n📈 Average Debt-to-Total Ratio: {debt_deals['debt_to_total_ratio'].mean():.1%}")
        
        # When is debt used?
        print(f"\n🔍 WHEN IS DEBT USED?")
        print(f"{'='*70}")
        
        # By industry
        print(f"\n🏭 Top Industries for Debt Deals:")
        debt_by_industry = debt_deals['Industry'].value_counts().head(5)
        for industry, count in debt_by_industry.items():
            pct = (count / len(debt_deals)) * 100
            print(f"   {industry:30s}: {count:2d} deals ({pct:5.1f}%)")
        
        # By revenue category
        if 'revenue_category' in debt_deals.columns:
            print(f"\n💵 Debt Deals by Revenue Category:")
            debt_by_revenue = debt_deals['revenue_category'].value_counts()
            for category, count in debt_by_revenue.items():
                print(f"   {str(category):20s}: {count:2d} deals")
        
        # By shark
        print(f"\n🦈 Sharks Who Use Debt Most:")
        shark_debt_counts = {}
        for shark in sharks:
            debt_col = f'{shark} Debt Amount'
            if debt_col in debt_deals.columns:
                shark_debt = debt_deals[debt_col].notna().sum()
                if shark_debt > 0:
                    shark_debt_counts[shark] = shark_debt
        
        for shark, count in sorted(shark_debt_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {shark:10s}: {count:2d} deals")
        
        # Characteristics comparison
        print(f"\n📊 DEBT DEAL CHARACTERISTICS")
        print(f"{'='*70}")
        
        pure_equity = df[df['deal_structure'] == 'Pure Equity'].copy()
        
        comparison_metrics = {
            'Avg Revenue': ('Yearly Revenue', 'median'),
            'Avg Gross Margin': ('Gross Margin', 'mean'),
            'Avg Net Margin': ('Net Margin', 'mean')
        }
        
        print(f"\n{'Metric':<20s} {'Debt Deals':>15s} {'Pure Equity':>15s} {'Difference':>15s}")
        print("-" * 70)
        
        for metric_name, (col, agg_func) in comparison_metrics.items():
            if col in debt_deals.columns and col in pure_equity.columns:
                if agg_func == 'mean':
                    debt_val = debt_deals[col].mean()
                    equity_val = pure_equity[col].mean()
                else:
                    debt_val = debt_deals[col].median()
                    equity_val = pure_equity[col].median()
                
                if pd.notna(debt_val) and pd.notna(equity_val) and equity_val != 0:
                    diff_pct = ((debt_val - equity_val) / equity_val) * 100
                    print(f"{metric_name:<20s} {debt_val:>15.1f} {equity_val:>15.1f} {diff_pct:>14.1f}%")
else:
    print("\n⚠️  No debt data available in dataset")

# ============================================================================
# 8.3 ROYALTY DEAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("8.3 ROYALTY DEAL ANALYSIS")
print("="*80)

# Analyze royalty deals
if 'Royalty Percentage' in df.columns:
    royalty_deals = df[(df['Royalty Percentage'] > 0) & pd.notna(df['Royalty Percentage'])].copy()
    
    print(f"\n👑 ROYALTY DEALS OVERVIEW")
    print(f"{'='*70}")
    print(f"Total Royalty Deals: {len(royalty_deals)}")
    print(f"Percentage of all deals: {len(royalty_deals)/total_funded*100:.1f}%")
    
    if len(royalty_deals) > 0:
        # Royalty characteristics
        print(f"\n📊 Royalty Percentage Statistics:")
        print(f"   Mean: {royalty_deals['Royalty Percentage'].mean():.2f}%")
        print(f"   Median: {royalty_deals['Royalty Percentage'].median():.2f}%")
        print(f"   Range: {royalty_deals['Royalty Percentage'].min():.1f}% - {royalty_deals['Royalty Percentage'].max():.1f}%")
        
        # Recoupment amounts
        if 'Royalty Recouped Amount' in royalty_deals.columns:
            recoup = royalty_deals[royalty_deals['Royalty Recouped Amount'] > 0]
            print(f"\n💰 Recoupment Amounts:")
            print(f"   Deals with recoupment clause: {len(recoup)}")
            if len(recoup) > 0:
                print(f"   Avg recoupment: ₹{recoup['Royalty Recouped Amount'].mean():.0f}L")
                print(f"   Median recoupment: ₹{recoup['Royalty Recouped Amount'].median():.0f}L")
        
        # When is royalty used?
        print(f"\n🔍 WHEN IS ROYALTY USED?")
        print(f"{'='*70}")
        
        # By industry
        print(f"\n🏭 Top Industries for Royalty Deals:")
        royalty_by_industry = royalty_deals['Industry'].value_counts().head(5)
        for industry, count in royalty_by_industry.items():
            pct = (count / len(royalty_deals)) * 100
            print(f"   {industry:30s}: {count:2d} deals ({pct:5.1f}%)")
        
        # By revenue
        if 'revenue_category' in royalty_deals.columns:
            print(f"\n💵 Royalty Deals by Revenue Category:")
            royalty_by_revenue = royalty_deals['revenue_category'].value_counts()
            for category, count in royalty_by_revenue.items():
                print(f"   {str(category):20s}: {count:2d} deals")
        
        # By shark
        print(f"\n🦈 Sharks Who Use Royalty Most:")
        shark_royalty_counts = {}
        for shark in sharks:
            inv_col = f'{shark} Investment Amount'
            if inv_col in royalty_deals.columns:
                shark_royalty = len(royalty_deals[royalty_deals[inv_col] > 0])
                if shark_royalty > 0:
                    shark_royalty_counts[shark] = shark_royalty
        
        for shark, count in sorted(shark_royalty_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {shark:10s}: {count:2d} deals")
        
        # Royalty deal profile
        print(f"\n📊 ROYALTY DEAL PROFILE")
        print(f"{'='*70}")
        print(f"   Avg Royalty %: {royalty_deals['Royalty Percentage'].mean():.2f}%")
        print(f"   Avg Equity Given: {royalty_deals['Total Deal Equity'].mean():.2f}%")
        print(f"   Avg Deal Amount: ₹{royalty_deals['Total Deal Amount'].mean():.1f}L")
        print(f"   Median Revenue: ₹{royalty_deals['Yearly Revenue'].median():.0f}L")
else:
    print("\n⚠️  No royalty data available in dataset")

# ============================================================================
# 8.4 PREDICTIVE MODEL: COMPLEX DEAL TERMS
# ============================================================================
print("\n" + "="*80)
print("8.4 PREDICTIVE MODEL FOR COMPLEX DEAL TERMS")
print("="*80)

# Create binary target: complex (1) vs simple (0)
df['is_complex_deal'] = df['deal_structure'].isin([
    'Debt + Equity', 
    'Royalty + Equity', 
    'Debt + Royalty + Equity'
]).astype(int)

print(f"\n🎯 Building model to predict complex deal structures...")

# Prepare data (only funded startups)
funded_df = df[df['Received Offer'] == 1].copy()

# Features for prediction
base_features = [
    'Yearly Revenue',
    'Monthly Sales',
    'Gross Margin',
    'Net Margin',
    'Original Ask Amount',
    'Valuation Requested'
]

# Add categorical features if they exist
categorical_features = []
if 'revenue_category' in funded_df.columns:
    categorical_features.append('revenue_category')
if 'Industry' in funded_df.columns:
    categorical_features.append('Industry')
if 'Cash Burn' in funded_df.columns:
    categorical_features.append('Cash Burn')

# Filter to available features
available_features = [f for f in base_features if f in funded_df.columns]
complex_features = available_features + categorical_features

print(f"   Features used: {len(complex_features)}")
print(f"   Training samples: {len(funded_df)}")
print(f"   Complex deals: {funded_df['is_complex_deal'].sum()} ({funded_df['is_complex_deal'].mean()*100:.1f}%)")

# Prepare data
X_complex = funded_df[complex_features].copy()
y_complex = funded_df['is_complex_deal']

# Handle missing values
for col in available_features:
    X_complex[col].fillna(X_complex[col].median(), inplace=True)

# Encode categoricals
X_complex_encoded = pd.get_dummies(X_complex, columns=categorical_features, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_complex_encoded, y_complex, test_size=0.2, random_state=42, stratify=y_complex
)

# Train decision tree for interpretability
dt_complex = DecisionTreeClassifier(
    max_depth=4, 
    min_samples_leaf=15, 
    min_samples_split=30,
    random_state=42
)
dt_complex.fit(X_train, y_train)

# Evaluate
y_pred = dt_complex.predict(X_test)
print(f"\n📊 Model Performance:")
print(f"   Training Accuracy: {dt_complex.score(X_train, y_train):.3f}")
print(f"   Test Accuracy: {dt_complex.score(X_test, y_test):.3f}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Simple', 'Complex']))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_complex_encoded.columns,
    'importance': dt_complex.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔝 Top 10 Features for Complex Deals:")
for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:40s}: {row['importance']:.4f}")

# Save model
with open('models/deal_structure_predictor.pkl', 'wb') as f:
    pickle.dump({
        'model': dt_complex,
        'feature_names': X_complex_encoded.columns.tolist(),
        'feature_importance': feature_importance
    }, f)
print(f"\n✅ Saved: models/deal_structure_predictor.pkl")

# Visualize decision tree
plt.figure(figsize=(24, 16))
tree.plot_tree(
    dt_complex,
    feature_names=X_complex_encoded.columns,
    class_names=['Simple', 'Complex'],
    filled=True,
    fontsize=9,
    rounded=True,
    proportion=True
)
plt.title('Decision Tree: When Are Complex Deal Terms Used?', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('reports/figures/complex_deal_decision_tree.png', dpi=300, bbox_inches='tight')
print("✅ Saved: reports/figures/complex_deal_decision_tree.png")
plt.close()

# Extract and display rules
def extract_rules(tree_model, feature_names, class_names):
    """Extract human-readable rules from decision tree"""
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree_.feature
    ]
    
    rules = []
    
    def recurse(node, path, depth=0):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left child (<=)
            recurse(tree_.children_left[node], 
                   path + [(name, '<=', threshold)], depth + 1)
            
            # Right child (>)
            recurse(tree_.children_right[node], 
                   path + [(name, '>', threshold)], depth + 1)
        else:
            # Leaf node
            class_pred = np.argmax(tree_.value[node])
            samples = tree_.value[node][0]
            confidence = samples[class_pred] / samples.sum()
            
            if class_pred == 1 and confidence > 0.5:  # Complex deal with confidence
                rules.append({
                    'path': path,
                    'class': class_names[class_pred],
                    'confidence': confidence,
                    'samples': int(samples.sum())
                })
    
    recurse(0, [])
    return rules

rules = extract_rules(dt_complex, X_complex_encoded.columns.tolist(), ['Simple', 'Complex'])

print(f"\n🎯 RULES FOR COMPLEX DEALS")
print(f"{'='*70}")
for i, rule in enumerate(rules, 1):
    print(f"\nRule {i} (Confidence: {rule['confidence']:.1%}, Samples: {rule['samples']}):")
    for feature, operator, threshold in rule['path']:
        print(f"   • {feature} {operator} {threshold:.2f}")

# ============================================================================
# 8.5 ADVISORY SHARES & SPECIAL TERMS
# ============================================================================
print("\n" + "="*80)
print("8.5 ADVISORY SHARES & SPECIAL TERMS")
print("="*80)

# Analyze advisory share deals
if 'Advisory Shares Equity' in df.columns:
    advisory_deals = df[(df['Advisory Shares Equity'] > 0) & pd.notna(df['Advisory Shares Equity'])].copy()
    
    print(f"\n🎓 ADVISORY SHARES ANALYSIS")
    print(f"{'='*70}")
    print(f"Total Advisory Deals: {len(advisory_deals)}")
    
    if len(advisory_deals) > 0:
        print(f"\n🏭 Industries:")
        for industry in advisory_deals['Industry'].unique():
            count = len(advisory_deals[advisory_deals['Industry'] == industry])
            print(f"   {industry}: {count} deal(s)")
        
        print(f"\n🦈 Sharks Involved:")
        for shark in sharks:
            inv_col = f'{shark} Investment Amount'
            if inv_col in advisory_deals.columns:
                count = len(advisory_deals[advisory_deals[inv_col] > 0])
                if count > 0:
                    print(f"   {shark}: {count} deal(s)")
    else:
        print("   No advisory share deals found")
else:
    print("\n⚠️  No advisory shares data available")

# Analyze conditional deals
if 'Deal Has Conditions' in df.columns:
    condition_deals = df[df['Deal Has Conditions'] == 'yes'].copy()
    
    print(f"\n⚖️  CONDITIONAL DEALS")
    print(f"{'='*70}")
    print(f"Total Conditional Deals: {len(condition_deals)}")
    if len(condition_deals) > 0:
        print(f"Percentage: {len(condition_deals)/total_funded*100:.1f}%")
        
        print(f"\n🏭 Industries with Most Conditions:")
        cond_by_industry = condition_deals['Industry'].value_counts().head(5)
        for industry, count in cond_by_industry.items():
            print(f"   {industry}: {count} deal(s)")
else:
    print("\n⚠️  No conditional deal data available")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("📊 DEAL STRUCTURE SUMMARY")
print("="*80)

print(f"\n📋 Overall Statistics:")
print(f"   Total Funded Deals: {total_funded}")
print(f"   Pure Equity: {structure_counts.get('Pure Equity', 0)} ({structure_counts.get('Pure Equity', 0)/total_funded*100:.1f}%)")
print(f"   Complex Deals: {funded_df['is_complex_deal'].sum()} ({funded_df['is_complex_deal'].mean()*100:.1f}%)")

if 'Total Deal Debt' in df.columns and len(debt_deals) > 0:
    print(f"\n💰 Debt Deals:")
    print(f"   Count: {len(debt_deals)}")
    print(f"   Avg Amount: ₹{debt_deals['Total Deal Debt'].mean():.1f}L")
    print(f"   Avg Ratio: {debt_deals['debt_to_total_ratio'].mean():.1%}")

if 'Royalty Percentage' in df.columns and len(royalty_deals) > 0:
    print(f"\n👑 Royalty Deals:")
    print(f"   Count: {len(royalty_deals)}")
    print(f"   Avg Royalty: {royalty_deals['Royalty Percentage'].mean():.2f}%")
    print(f"   Avg Equity: {royalty_deals['Total Deal Equity'].mean():.2f}%")

print(f"\n💡 Key Insights:")
print(f"   • {structure_counts.get('Pure Equity', 0)/total_funded*100:.1f}% of deals are pure equity")
print(f"   • Complex terms used in {funded_df['is_complex_deal'].mean()*100:.1f}% of deals")
print(f"   • Decision tree model accuracy: {dt_complex.score(X_test, y_test):.1%}")

print("\n" + "="*80)
print("✅ PHASE 8.1-8.5 COMPLETE")
print("="*80)
print("\nNext: Building Deal Structure Recommendation Engine (8.6)")
