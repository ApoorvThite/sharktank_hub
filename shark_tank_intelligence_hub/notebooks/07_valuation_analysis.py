"""
PHASE 5: VALUATION REALITY CHECK ANALYSIS
Complete valuation framework with industry benchmarks, fairness scoring, and calculator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("💰 PHASE 5: VALUATION REALITY CHECK ANALYSIS")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading processed data...")
df = pd.read_csv('data/processed/processed_data_full.csv')

print(f"✅ Loaded {len(df)} startups")
print(f"   Columns: {len(df.columns)}")
print(f"   Seasons: {df['Season Number'].min()} to {df['Season Number'].max()}")

# ============================================================================
# 5.1 INDUSTRY VALUATION BENCHMARKS
# ============================================================================
print("\n" + "="*80)
print("5.1 INDUSTRY VALUATION BENCHMARKS")
print("="*80)

# Calculate revenue multiples by industry
industry_benchmarks = df[df['Yearly Revenue'] > 0].groupby('Industry').agg({
    'revenue_multiple': ['mean', 'median', 'std', 'min', 'max', 'count'],
    'Total Deal Equity': ['mean', 'median'],
    'Valuation Requested': ['mean', 'median'],
    'got_offer': 'mean'
}).round(2)

# Rename columns for clarity
industry_benchmarks.columns = ['_'.join(col).strip() for col in industry_benchmarks.columns.values]

print("\n📊 Industry Benchmarks (Revenue-Generating Startups Only):")
print(industry_benchmarks.sort_values('revenue_multiple_median', ascending=False))

# Save benchmarks
industry_benchmarks.to_csv('reports/industry_benchmarks.csv')
print("\n✅ Saved: reports/industry_benchmarks.csv")

# Visualize revenue multiples by industry
fig, ax = plt.subplots(figsize=(14, 8))

# Filter industries with at least 5 startups
valid_industries = industry_benchmarks[industry_benchmarks['revenue_multiple_count'] >= 5].index
df_plot = df[(df['Yearly Revenue'] > 0) & (df['Industry'].isin(valid_industries))]

# Create box plot
box_data = []
labels = []
for industry in df_plot['Industry'].value_counts().head(10).index:
    data = df_plot[df_plot['Industry'] == industry]['revenue_multiple'].dropna()
    if len(data) >= 3:  # Need at least 3 points for box plot
        box_data.append(data)
        labels.append(industry)

bp = ax.boxplot(box_data, labels=labels, patch_artist=True)

# Color boxes
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

ax.set_xlabel('Industry', fontsize=12, fontweight='bold')
ax.set_ylabel('Revenue Multiple (Valuation / Revenue)', fontsize=12, fontweight='bold')
ax.set_title('Revenue Multiple Distribution by Industry', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('reports/figures/revenue_multiple_by_industry.png', dpi=300, bbox_inches='tight')
print("✅ Saved: reports/figures/revenue_multiple_by_industry.png")
plt.close()

# ============================================================================
# 5.2 VALUATION INFLATION SCORE
# ============================================================================
print("\n" + "="*80)
print("5.2 VALUATION INFLATION SCORE")
print("="*80)

def calculate_inflation_score(row):
    """
    Calculate how overvalued/undervalued a startup is
    
    Score interpretation:
    < -1.0: Significantly undervalued
    -1.0 to 0: Below market (shark-friendly)
    0 to +1.0: Fair market
    +1.0 to +2.0: Above market
    > +2.0: Significantly overvalued
    """
    if pd.isna(row['revenue_multiple']) or row['Yearly Revenue'] <= 0:
        return np.nan
    
    industry = row['Industry']
    asked_multiple = row['revenue_multiple']
    
    # Get industry statistics
    try:
        industry_median = industry_benchmarks.loc[industry, 'revenue_multiple_median']
        industry_std = industry_benchmarks.loc[industry, 'revenue_multiple_std']
        
        if pd.isna(industry_std) or industry_std == 0:
            return 0.0
        
        # Calculate z-score
        inflation_score = (asked_multiple - industry_median) / industry_std
        return inflation_score
    except:
        return np.nan

df['inflation_score'] = df.apply(calculate_inflation_score, axis=1)

print(f"\n📊 Inflation Score Statistics:")
print(df['inflation_score'].describe())

# Analyze success rates by inflation score
df['inflation_category'] = pd.cut(
    df['inflation_score'], 
    bins=[-np.inf, -1, 0, 1, 2, np.inf],
    labels=['Significantly Undervalued', 'Below Market', 'Fair Market', 'Above Market', 'Significantly Overvalued']
)

inflation_success = df.groupby('inflation_category').agg({
    'got_offer': ['mean', 'count']
}).round(3)

print("\n📊 Success Rate by Valuation Inflation:")
print(inflation_success)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Success rate by inflation category
success_data = df.groupby('inflation_category')['got_offer'].mean() * 100
success_data.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7)
ax1.set_xlabel('Valuation Category', fontsize=12, fontweight='bold')
ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Deal Success Rate by Valuation Inflation', fontsize=14, fontweight='bold')
ax1.axhline(y=df['got_offer'].mean()*100, color='red', linestyle='--', label='Overall Average')
ax1.legend()
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Distribution of inflation scores
df['inflation_score'].dropna().hist(bins=30, ax=ax2, color='coral', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Inflation Score (Z-Score)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Distribution of Valuation Inflation Scores', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Fair Market')
ax2.axvline(x=1, color='orange', linestyle='--', linewidth=2, label='Above Market')
ax2.axvline(x=-1, color='blue', linestyle='--', linewidth=2, label='Below Market')
ax2.legend()

plt.tight_layout()
plt.savefig('reports/figures/valuation_inflation_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: reports/figures/valuation_inflation_analysis.png")
plt.close()

# ============================================================================
# 5.3 EQUITY DILUTION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5.3 EQUITY DILUTION ANALYSIS")
print("="*80)

# Calculate equity dilution for deals that happened
df['equity_dilution'] = df['Total Deal Equity'] - df['ask_percentage']

# Analyze by revenue category
dilution_analysis = df[df['got_offer'] == 1].groupby('revenue_category').agg({
    'ask_percentage': ['mean', 'median', 'std'],
    'Total Deal Equity': ['mean', 'median', 'std'],
    'equity_dilution': ['mean', 'median', 'std', 'count']
}).round(2)

print("\n📊 Equity Dilution by Revenue Category:")
print(dilution_analysis)

# Save dilution analysis
dilution_analysis.to_csv('reports/equity_dilution_analysis.csv')
print("\n✅ Saved: reports/equity_dilution_analysis.csv")

# Visualize equity dilution
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Asked vs Given Equity by Revenue Category
ax1 = axes[0, 0]
dilution_summary = df[df['got_offer'] == 1].groupby('revenue_category').agg({
    'ask_percentage': 'mean',
    'Total Deal Equity': 'mean'
}).round(2)

x = np.arange(len(dilution_summary.index))
width = 0.35

ax1.bar(x - width/2, dilution_summary['ask_percentage'], width, label='Asked', color='lightblue', alpha=0.8)
ax1.bar(x + width/2, dilution_summary['Total Deal Equity'], width, label='Given', color='coral', alpha=0.8)

ax1.set_xlabel('Revenue Category', fontsize=11, fontweight='bold')
ax1.set_ylabel('Equity %', fontsize=11, fontweight='bold')
ax1.set_title('Asked vs Given Equity by Revenue Segment', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(dilution_summary.index, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Equity Dilution Distribution
ax2 = axes[0, 1]
df[df['got_offer'] == 1]['equity_dilution'].hist(bins=30, ax=ax2, color='purple', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Equity Dilution (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Distribution of Equity Dilution', fontsize=12, fontweight='bold')
ax2.axvline(x=0, color='green', linestyle='--', linewidth=2, label='No Dilution')
ax2.axvline(x=df[df['got_offer'] == 1]['equity_dilution'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {df[df["got_offer"] == 1]["equity_dilution"].mean():.1f}%')
ax2.legend()

# 3. Dilution by Industry (top 8)
ax3 = axes[1, 0]
top_industries = df[df['got_offer'] == 1]['Industry'].value_counts().head(8).index
industry_dilution = df[(df['got_offer'] == 1) & (df['Industry'].isin(top_industries))].groupby('Industry')['equity_dilution'].mean().sort_values()

industry_dilution.plot(kind='barh', ax=ax3, color='teal', alpha=0.7)
ax3.set_xlabel('Average Equity Dilution (%)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Industry', fontsize=11, fontweight='bold')
ax3.set_title('Average Equity Dilution by Industry (Top 8)', fontsize=12, fontweight='bold')
ax3.axvline(x=0, color='green', linestyle='--', linewidth=2)
ax3.grid(axis='x', alpha=0.3)

# 4. Dilution vs Revenue (scatter)
ax4 = axes[1, 1]
scatter_data = df[(df['got_offer'] == 1) & (df['Yearly Revenue'] > 0) & (df['Yearly Revenue'] < 10000)]
ax4.scatter(scatter_data['Yearly Revenue'], scatter_data['equity_dilution'], 
            alpha=0.5, c=scatter_data['Total Deal Amount'], cmap='viridis', s=50)
ax4.set_xlabel('Yearly Revenue (₹L)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Equity Dilution (%)', fontsize=11, fontweight='bold')
ax4.set_title('Equity Dilution vs Revenue (colored by Deal Amount)', fontsize=12, fontweight='bold')
ax4.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5)
cbar = plt.colorbar(ax4.collections[0], ax=ax4)
cbar.set_label('Deal Amount (₹L)', fontsize=10)

plt.tight_layout()
plt.savefig('reports/figures/equity_dilution_by_revenue.png', dpi=300, bbox_inches='tight')
print("✅ Saved: reports/figures/equity_dilution_by_revenue.png")
plt.close()

# ============================================================================
# 5.4 DEAL FAIRNESS INDEX
# ============================================================================
print("\n" + "="*80)
print("5.4 DEAL FAIRNESS INDEX")
print("="*80)

def calculate_fairness_index(row):
    """
    Composite score of deal fairness (0-100)
    
    Components:
    1. Valuation Reasonableness (40%): Deviation from industry median
    2. Equity Ask Fairness (30%): Appropriate for stage/revenue
    3. Deal Terms Complexity (30%): Penalty for debt/royalty
    """
    # Component 1: Valuation score (0-100)
    inflation = row['inflation_score']
    if pd.isna(inflation):
        valuation_score = 70  # Neutral for missing data
    elif inflation < -1:
        valuation_score = 60  # Too low may indicate issues
    elif inflation < 0:
        valuation_score = 100  # Perfect
    elif inflation < 1:
        valuation_score = 85  # Good
    elif inflation < 2:
        valuation_score = 60  # Above market
    else:
        valuation_score = 30  # Overvalued
    
    # Component 2: Equity score (0-100)
    revenue = row['Yearly Revenue']
    equity_asked = row['ask_percentage']
    
    # Expected equity by revenue
    if revenue == 0:
        expected_equity = 12
    elif revenue < 100:
        expected_equity = 9
    elif revenue < 1000:
        expected_equity = 7
    else:
        expected_equity = 5
    
    equity_deviation = abs(equity_asked - expected_equity)
    equity_score = max(0, 100 - (equity_deviation * 10))
    
    # Component 3: Terms score (0-100)
    has_debt = row['Total Deal Debt'] > 0 if pd.notna(row['Total Deal Debt']) else False
    has_royalty = row['Royalty Percentage'] > 0 if pd.notna(row['Royalty Percentage']) else False
    
    terms_score = 100
    if has_debt:
        terms_score -= 20
    if has_royalty:
        terms_score -= 15
    
    # Weighted average
    fairness_index = (
        0.4 * valuation_score +
        0.3 * equity_score +
        0.3 * terms_score
    )
    
    return fairness_index

df['fairness_index'] = df.apply(calculate_fairness_index, axis=1)

# Categorize fairness
df['fairness_category'] = pd.cut(
    df['fairness_index'],
    bins=[0, 40, 60, 80, 100],
    labels=['Unfair', 'Moderate', 'Fair', 'Highly Fair']
)

print("\n📊 Deal Fairness Distribution:")
fairness_dist = df['fairness_category'].value_counts(normalize=True).sort_index() * 100
print(fairness_dist.round(1))

# Analyze success by fairness
fairness_success = df.groupby('fairness_category').agg({
    'got_offer': ['mean', 'count']
}).round(3)

print("\n📊 Success Rate by Fairness Category:")
print(fairness_success)

# Visualize fairness index
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Fairness distribution
fairness_dist.plot(kind='bar', ax=ax1, color=['red', 'orange', 'lightgreen', 'darkgreen'], alpha=0.7)
ax1.set_xlabel('Fairness Category', fontsize=12, fontweight='bold')
ax1.set_ylabel('Percentage of Startups (%)', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Deal Fairness Index', fontsize=14, fontweight='bold')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.grid(axis='y', alpha=0.3)

# Success rate by fairness
success_by_fairness = df.groupby('fairness_category')['got_offer'].mean() * 100
success_by_fairness.plot(kind='bar', ax=ax2, color='steelblue', alpha=0.7)
ax2.set_xlabel('Fairness Category', fontsize=12, fontweight='bold')
ax2.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Deal Success Rate by Fairness Index', fontsize=14, fontweight='bold')
ax2.axhline(y=df['got_offer'].mean()*100, color='red', linestyle='--', label='Overall Average')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/deal_fairness_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: reports/figures/deal_fairness_analysis.png")
plt.close()

# ============================================================================
# SAVE ENHANCED DATASET
# ============================================================================
print("\n" + "="*80)
print("💾 SAVING ENHANCED DATASET")
print("="*80)

# Save enhanced dataset with new features
df.to_csv('data/processed/processed_data_with_valuation_metrics.csv', index=False)
print("✅ Saved: data/processed/processed_data_with_valuation_metrics.csv")

# Export key metrics
valuation_metrics = df[[
    'Startup Name', 'Industry', 'Yearly Revenue', 'Valuation Requested',
    'revenue_multiple', 'inflation_score', 'inflation_category',
    'ask_percentage', 'Total Deal Equity', 'equity_dilution',
    'fairness_index', 'fairness_category', 'got_offer'
]].copy()

valuation_metrics.to_csv('reports/valuation_metrics_summary.csv', index=False)
print("✅ Saved: reports/valuation_metrics_summary.csv")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("📊 VALUATION ANALYSIS SUMMARY")
print("="*80)

print(f"\n🎯 Key Findings:")
print(f"   • Total startups analyzed: {len(df)}")
print(f"   • Revenue-generating startups: {(df['Yearly Revenue'] > 0).sum()}")
print(f"   • Average revenue multiple: {df['revenue_multiple'].mean():.2f}x")
print(f"   • Median revenue multiple: {df['revenue_multiple'].median():.2f}x")

print(f"\n💰 Valuation Inflation:")
print(f"   • Fair market (±1 std): {((df['inflation_score'] >= -1) & (df['inflation_score'] <= 1)).sum()} startups")
print(f"   • Overvalued (>1 std): {(df['inflation_score'] > 1).sum()} startups")
print(f"   • Undervalued (<-1 std): {(df['inflation_score'] < -1).sum()} startups")

print(f"\n📊 Equity Dilution (for deals):")
deals = df[df['got_offer'] == 1]
print(f"   • Average asked equity: {deals['ask_percentage'].mean():.2f}%")
print(f"   • Average given equity: {deals['Total Deal Equity'].mean():.2f}%")
print(f"   • Average dilution: {deals['equity_dilution'].mean():.2f}%")

print(f"\n⚖️ Deal Fairness:")
for category in ['Highly Fair', 'Fair', 'Moderate', 'Unfair']:
    count = (df['fairness_category'] == category).sum()
    pct = count / len(df) * 100
    print(f"   • {category}: {count} ({pct:.1f}%)")

print("\n" + "="*80)
print("✅ PHASE 5.1-5.4 COMPLETE")
print("="*80)
print("\nNext: Building Valuation Calculator Tool (5.5-5.6)")
