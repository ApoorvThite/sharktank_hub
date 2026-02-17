"""
PHASE 9: GEOGRAPHIC SUCCESS MAP & INTEGRATION
Comprehensive geographic analysis of startup success patterns by state and region
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("="*80)
print("🗺️ PHASE 9: GEOGRAPHIC SUCCESS MAP & INTEGRATION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading processed data...")
df = pd.read_csv('data/processed/processed_data_full.csv')

print(f"✅ Loaded {len(df)} startups")
print(f"   Deals: {df['got_offer'].sum()}")
print(f"   States: {df['Pitchers State'].nunique()}")

# ============================================================================
# 9.1 GEOGRAPHIC ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("9.1 STATE-WISE SUCCESS ANALYSIS")
print("="*80)

# State-wise statistics
state_stats = df.groupby('Pitchers State').agg({
    'Startup Name': 'count',
    'Received Offer': 'mean',
    'Total Deal Amount': 'mean',
    'Total Deal Equity': 'mean',
    'Valuation Requested': 'median',
    'Yearly Revenue': 'median'
}).round(2)

state_stats.columns = ['Total_Pitches', 'Offer_Rate', 'Avg_Deal', 'Avg_Equity', 'Median_Valuation', 'Median_Revenue']
state_stats = state_stats.sort_values('Total_Pitches', ascending=False)

# Filter states with at least 5 pitches for meaningful analysis
significant_states = state_stats[state_stats['Total_Pitches'] >= 5]

print(f"\n📍 Top 15 States by Pitch Volume:")
print(f"{'State':<25s} {'Pitches':>8s} {'Offer Rate':>12s} {'Avg Deal':>12s} {'Med Valuation':>15s}")
print("-" * 80)
for state, row in significant_states.head(15).iterrows():
    print(f"{state:<25s} {int(row['Total_Pitches']):>8d} {row['Offer_Rate']:>11.1%} "
          f"₹{row['Avg_Deal']:>10.1f}L ₹{row['Median_Valuation']:>13.0f}L")

# Save state statistics
state_stats.to_csv('reports/state_statistics.csv')
print("\n✅ Saved: reports/state_statistics.csv")

# Metro vs non-metro analysis
print(f"\n🏙️ METRO VS NON-METRO COMPARISON")
print("="*80)

metro_states = ['Maharashtra', 'Delhi', 'Karnataka', 'Telangana', 'Tamil Nadu', 'Gujarat']
df['is_metro'] = df['Pitchers State'].isin(metro_states).astype(int)

metro_comparison = df.groupby('is_metro').agg({
    'Startup Name': 'count',
    'Received Offer': 'mean',
    'Total Deal Amount': 'mean',
    'Valuation Requested': 'median',
    'Yearly Revenue': 'median',
    'Gross Margin': 'mean'
}).round(2)

metro_comparison.columns = ['Count', 'Offer_Rate', 'Avg_Deal', 'Median_Valuation', 'Median_Revenue', 'Avg_Margin']

print(f"\n{'Metric':<25s} {'Metro':>15s} {'Non-Metro':>15s} {'Difference':>15s}")
print("-" * 80)
for metric in metro_comparison.columns:
    metro_val = metro_comparison.loc[1, metric]
    non_metro_val = metro_comparison.loc[0, metric]
    
    if metric == 'Count':
        diff = metro_val - non_metro_val
        print(f"{metric:<25s} {metro_val:>15.0f} {non_metro_val:>15.0f} {diff:>15.0f}")
    elif metric in ['Offer_Rate', 'Avg_Margin']:
        diff_pct = ((metro_val - non_metro_val) / non_metro_val * 100) if non_metro_val != 0 else 0
        print(f"{metric:<25s} {metro_val:>14.1%} {non_metro_val:>14.1%} {diff_pct:>14.1f}%")
    else:
        diff_pct = ((metro_val - non_metro_val) / non_metro_val * 100) if non_metro_val != 0 else 0
        print(f"{metric:<25s} ₹{metro_val:>13.1f}L ₹{non_metro_val:>13.1f}L {diff_pct:>14.1f}%")

# Industry concentration by state
print(f"\n🏭 STATE INDUSTRY SPECIALIZATIONS")
print("="*80)

# Get top states
top_states = significant_states.head(10).index

state_industry = pd.crosstab(
    df[df['Pitchers State'].isin(top_states)]['Pitchers State'],
    df[df['Pitchers State'].isin(top_states)]['Industry'],
    normalize='index'
) * 100

print(f"\n{'State':<25s} {'Top Industry':<30s} {'Concentration':>15s}")
print("-" * 80)
for state in top_states:
    if state in state_industry.index:
        top_industry = state_industry.loc[state].idxmax()
        pct = state_industry.loc[state].max()
        print(f"{state:<25s} {top_industry:<30s} {pct:>14.1f}%")

# Save industry concentration
state_industry.to_csv('reports/state_industry_concentration.csv')
print("\n✅ Saved: reports/state_industry_concentration.csv")

# Regional analysis
print(f"\n🌏 REGIONAL ANALYSIS")
print("="*80)

# Define regions
region_mapping = {
    'Maharashtra': 'West',
    'Gujarat': 'West',
    'Goa': 'West',
    'Rajasthan': 'West',
    'Delhi': 'North',
    'Haryana': 'North',
    'Punjab': 'North',
    'Uttar Pradesh': 'North',
    'Himachal Pradesh': 'North',
    'Jammu and Kashmir': 'North',
    'Uttarakhand': 'North',
    'Karnataka': 'South',
    'Tamil Nadu': 'South',
    'Telangana': 'South',
    'Andhra Pradesh': 'South',
    'Kerala': 'South',
    'West Bengal': 'East',
    'Odisha': 'East',
    'Bihar': 'East',
    'Jharkhand': 'East',
    'Assam': 'Northeast',
    'Meghalaya': 'Northeast',
    'Madhya Pradesh': 'Central',
    'Chhattisgarh': 'Central'
}

df['Region'] = df['Pitchers State'].map(region_mapping).fillna('Other')

regional_stats = df.groupby('Region').agg({
    'Startup Name': 'count',
    'Received Offer': 'mean',
    'Total Deal Amount': 'mean',
    'Valuation Requested': 'median'
}).round(2)

regional_stats.columns = ['Pitches', 'Offer_Rate', 'Avg_Deal', 'Median_Valuation']
regional_stats = regional_stats.sort_values('Pitches', ascending=False)

print(f"\n{'Region':<15s} {'Pitches':>10s} {'Offer Rate':>12s} {'Avg Deal':>12s}")
print("-" * 60)
for region, row in regional_stats.iterrows():
    print(f"{region:<15s} {int(row['Pitches']):>10d} {row['Offer_Rate']:>11.1%} ₹{row['Avg_Deal']:>10.1f}L")

# ============================================================================
# 9.2 GEOGRAPHIC VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("9.2 GEOGRAPHIC VISUALIZATIONS")
print("="*80)

# Visualization 1: Top states bar chart
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Plot 1: Pitch volume by state
top_15_states = significant_states.head(15)
axes[0, 0].barh(range(len(top_15_states)), top_15_states['Total_Pitches'], color='steelblue', alpha=0.8)
axes[0, 0].set_yticks(range(len(top_15_states)))
axes[0, 0].set_yticklabels(top_15_states.index)
axes[0, 0].set_xlabel('Number of Pitches', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Top 15 States by Pitch Volume', fontsize=14, fontweight='bold', pad=15)
axes[0, 0].invert_yaxis()
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Success rate by state
success_sorted = significant_states.sort_values('Offer_Rate', ascending=True).tail(15)
colors = ['green' if x > 0.66 else 'orange' if x > 0.5 else 'red' for x in success_sorted['Offer_Rate']]
axes[0, 1].barh(range(len(success_sorted)), success_sorted['Offer_Rate'] * 100, color=colors, alpha=0.8)
axes[0, 1].set_yticks(range(len(success_sorted)))
axes[0, 1].set_yticklabels(success_sorted.index)
axes[0, 1].set_xlabel('Offer Rate (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Top 15 States by Success Rate', fontsize=14, fontweight='bold', pad=15)
axes[0, 1].axvline(66.2, color='red', linestyle='--', linewidth=2, label='Overall Avg')
axes[0, 1].legend()
axes[0, 1].invert_yaxis()
axes[0, 1].grid(axis='x', alpha=0.3)

# Plot 3: Metro vs Non-Metro
categories = ['Metro', 'Non-Metro']
offer_rates = [metro_comparison.loc[1, 'Offer_Rate'] * 100, metro_comparison.loc[0, 'Offer_Rate'] * 100]
axes[1, 0].bar(categories, offer_rates, color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.6)
axes[1, 0].set_ylabel('Offer Rate (%)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Metro vs Non-Metro Success Rates', fontsize=14, fontweight='bold', pad=15)
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(offer_rates):
    axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)

# Plot 4: Regional distribution
regional_sorted = regional_stats.sort_values('Pitches', ascending=True)
axes[1, 1].barh(range(len(regional_sorted)), regional_sorted['Pitches'], color='coral', alpha=0.8)
axes[1, 1].set_yticks(range(len(regional_sorted)))
axes[1, 1].set_yticklabels(regional_sorted.index)
axes[1, 1].set_xlabel('Number of Pitches', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Pitch Distribution by Region', fontsize=14, fontweight='bold', pad=15)
axes[1, 1].invert_yaxis()
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/geographic_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: reports/figures/geographic_analysis.png")
plt.close()

# Create interactive visualizations with Plotly
print("\n📊 Creating interactive visualizations...")

# Interactive bar chart: Top states
fig_states = px.bar(
    significant_states.head(15).reset_index(),
    y='Pitchers State',
    x='Total_Pitches',
    orientation='h',
    title='Top 15 States by Startup Pitches',
    labels={'Total_Pitches': 'Number of Pitches', 'Pitchers State': 'State'},
    color='Offer_Rate',
    color_continuous_scale='Viridis',
    hover_data=['Offer_Rate', 'Avg_Deal', 'Median_Valuation']
)
fig_states.update_layout(height=600, showlegend=False)
fig_states.write_html('reports/figures/interactive_states.html')
print("✅ Saved: reports/figures/interactive_states.html")

# Scatter plot: Pitches vs Success Rate
fig_scatter = px.scatter(
    significant_states.reset_index(),
    x='Total_Pitches',
    y='Offer_Rate',
    size='Avg_Deal',
    color='Median_Valuation',
    hover_name='Pitchers State',
    title='State Ecosystem Analysis: Volume vs Success',
    labels={
        'Total_Pitches': 'Number of Pitches',
        'Offer_Rate': 'Success Rate',
        'Avg_Deal': 'Avg Deal Size (₹L)',
        'Median_Valuation': 'Median Valuation (₹L)'
    },
    color_continuous_scale='Plasma'
)
fig_scatter.update_layout(height=600)
fig_scatter.write_html('reports/figures/ecosystem_scatter.html')
print("✅ Saved: reports/figures/ecosystem_scatter.html")

# Sunburst chart: Region > State > Industry
# Prepare data for sunburst
sunburst_data = []
for _, row in df.iterrows():
    if pd.notna(row['Pitchers State']) and pd.notna(row['Industry']) and pd.notna(row['Region']):
        sunburst_data.append({
            'Region': row['Region'],
            'State': row['Pitchers State'],
            'Industry': row['Industry']
        })

sunburst_df = pd.DataFrame(sunburst_data)
sunburst_counts = sunburst_df.groupby(['Region', 'State', 'Industry']).size().reset_index(name='Count')

# Filter to top regions and states
top_regions = regional_stats.head(4).index
sunburst_filtered = sunburst_counts[sunburst_counts['Region'].isin(top_regions)]

fig_sunburst = px.sunburst(
    sunburst_filtered,
    path=['Region', 'State', 'Industry'],
    values='Count',
    title='Geographic & Industry Distribution Hierarchy',
    color='Count',
    color_continuous_scale='RdYlGn'
)
fig_sunburst.update_layout(height=700)
fig_sunburst.write_html('reports/figures/geographic_sunburst.html')
print("✅ Saved: reports/figures/geographic_sunburst.html")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("📊 GEOGRAPHIC ANALYSIS SUMMARY")
print("="*80)

print(f"\n🗺️ Coverage:")
print(f"   Total States/UTs: {df['Pitchers State'].nunique()}")
print(f"   States with 10+ pitches: {len(state_stats[state_stats['Total_Pitches'] >= 10])}")
print(f"   States with 5+ pitches: {len(significant_states)}")

print(f"\n🏆 Top Performers:")
print(f"   Most Pitches: {state_stats.index[0]} ({int(state_stats.iloc[0]['Total_Pitches'])} pitches)")
highest_success = significant_states.sort_values('Offer_Rate', ascending=False).iloc[0]
print(f"   Highest Success Rate: {significant_states.sort_values('Offer_Rate', ascending=False).index[0]} ({highest_success['Offer_Rate']:.1%})")
print(f"   Highest Avg Deal: {significant_states.sort_values('Avg_Deal', ascending=False).index[0]} (₹{significant_states.sort_values('Avg_Deal', ascending=False).iloc[0]['Avg_Deal']:.1f}L)")

print(f"\n🏙️ Metro Advantage:")
metro_pitches = df[df['is_metro'] == 1].shape[0]
non_metro_pitches = df[df['is_metro'] == 0].shape[0]
print(f"   Metro Pitches: {metro_pitches} ({metro_pitches/len(df)*100:.1f}%)")
print(f"   Non-Metro Pitches: {non_metro_pitches} ({non_metro_pitches/len(df)*100:.1f}%)")
print(f"   Metro Success Rate: {metro_comparison.loc[1, 'Offer_Rate']:.1%}")
print(f"   Non-Metro Success Rate: {metro_comparison.loc[0, 'Offer_Rate']:.1%}")
diff = ((metro_comparison.loc[1, 'Offer_Rate'] - metro_comparison.loc[0, 'Offer_Rate']) / 
        metro_comparison.loc[0, 'Offer_Rate'] * 100)
print(f"   Metro Advantage: {diff:+.1f}%")

print(f"\n🌏 Regional Distribution:")
for region, row in regional_stats.iterrows():
    pct = row['Pitches'] / len(df) * 100
    print(f"   {region:15s}: {int(row['Pitches']):3d} pitches ({pct:5.1f}%), {row['Offer_Rate']:.1%} success")

print(f"\n💡 Key Insights:")
print(f"   • Maharashtra dominates with {int(state_stats.iloc[0]['Total_Pitches'])} pitches")
print(f"   • Metro states have {diff:+.1f}% higher success rates")
print(f"   • West region accounts for {regional_stats.loc['West', 'Pitches']/len(df)*100:.1f}% of all pitches")
print(f"   • Top 5 states represent {state_stats.head(5)['Total_Pitches'].sum()/len(df)*100:.1f}% of ecosystem")

print("\n" + "="*80)
print("✅ PHASE 9.1-9.2 COMPLETE")
print("="*80)
print("\nNext: Building Integrated Streamlit Dashboard (9.3-9.4)")
