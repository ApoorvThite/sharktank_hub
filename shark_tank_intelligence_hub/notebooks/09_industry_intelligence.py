"""
PHASE 7: INDUSTRY DEEP DIVE DASHBOARD
Comprehensive industry analysis with profiles, success factors, trends, and entry requirements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("="*80)
print("🏭 PHASE 7: INDUSTRY DEEP DIVE DASHBOARD")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading processed data...")
df = pd.read_csv('data/processed/processed_data_full.csv')

# Load valuation metrics if available
try:
    df_valuation = pd.read_csv('data/processed/processed_data_with_valuation_metrics.csv')
    df = df_valuation
    print("✅ Loaded data with valuation metrics")
except:
    print("⚠️  Valuation metrics not found, using base data")

print(f"✅ Loaded {len(df)} startups")
print(f"   Deals: {df['got_offer'].sum()}")
print(f"   Industries: {df['Industry'].nunique()}")

# Define sharks
sharks = ['Namita', 'Aman', 'Anupam', 'Peyush', 'Vineeta', 'Ritesh', 'Amit']

# ============================================================================
# 7.1 INDUSTRY PROFILE GENERATOR
# ============================================================================
print("\n" + "="*80)
print("7.1 GENERATING COMPREHENSIVE INDUSTRY PROFILES")
print("="*80)

def generate_industry_profile(industry_name):
    """Create comprehensive profile for an industry"""
    industry_df = df[df['Industry'] == industry_name].copy()
    
    if len(industry_df) == 0:
        return None
    
    # Basic metrics
    total_pitches = len(industry_df)
    market_share = (total_pitches / len(df)) * 100
    
    # Success metrics
    offer_rate = industry_df['Received Offer'].mean() if 'Received Offer' in industry_df.columns else 0
    
    # Acceptance rate (of those who got offers)
    offers_received = industry_df[industry_df['Received Offer'] == 1]
    acceptance_rate = offers_received['Accepted Offer'].mean() if len(offers_received) > 0 and 'Accepted Offer' in industry_df.columns else 0
    
    # Financial metrics
    avg_revenue = industry_df['Yearly Revenue'].median()
    avg_gross_margin = industry_df['Gross Margin'].mean() if 'Gross Margin' in industry_df.columns else 0
    avg_net_margin = industry_df['Net Margin'].mean() if 'Net Margin' in industry_df.columns else 0
    
    # Deal terms
    avg_ask_amount = industry_df['Original Ask Amount'].median()
    avg_valuation = industry_df['Valuation Requested'].median()
    
    deals_df = industry_df[industry_df['Total Deal Amount'] > 0]
    avg_deal_size = deals_df['Total Deal Amount'].mean() if len(deals_df) > 0 else 0
    avg_equity_given = deals_df['Total Deal Equity'].mean() if len(deals_df) > 0 else 0
    
    # Revenue multiple
    if 'revenue_multiple' in industry_df.columns:
        revenue_multiple = industry_df[industry_df['revenue_multiple'] > 0]['revenue_multiple'].median()
    else:
        # Calculate it
        rev_mult_df = industry_df[(industry_df['Yearly Revenue'] > 0) & (industry_df['Valuation Requested'] > 0)]
        if len(rev_mult_df) > 0:
            revenue_multiple = (rev_mult_df['Valuation Requested'] / rev_mult_df['Yearly Revenue']).median()
        else:
            revenue_multiple = 0
    
    # Geographic concentration
    if 'Pitchers State' in industry_df.columns:
        top_states = industry_df['Pitchers State'].value_counts().head(3).to_dict()
    else:
        top_states = {}
    
    # Demographics
    avg_age_group = None
    if 'Pitchers Average Age' in industry_df.columns:
        mode_age = industry_df['Pitchers Average Age'].mode()
        avg_age_group = mode_age[0] if len(mode_age) > 0 else None
    
    female_founder_pct = 0
    if 'has_female_founder' in industry_df.columns:
        female_founder_pct = (industry_df['has_female_founder'].sum() / len(industry_df)) * 100
    
    # Innovation
    patent_rate = 0
    if 'has_patent' in industry_df.columns:
        patent_rate = industry_df['has_patent'].mean() * 100
    
    bootstrap_rate = 0
    if 'is_bootstrapped' in industry_df.columns:
        bootstrap_rate = industry_df['is_bootstrapped'].mean() * 100
    
    profile = {
        'industry': industry_name,
        'total_pitches': int(total_pitches),
        'market_share': float(market_share),
        'offer_rate': float(offer_rate),
        'acceptance_rate': float(acceptance_rate),
        'avg_revenue': float(avg_revenue),
        'avg_gross_margin': float(avg_gross_margin),
        'avg_net_margin': float(avg_net_margin),
        'avg_ask_amount': float(avg_ask_amount),
        'avg_valuation': float(avg_valuation),
        'avg_deal_size': float(avg_deal_size),
        'avg_equity_given': float(avg_equity_given),
        'revenue_multiple': float(revenue_multiple),
        'top_states': top_states,
        'avg_age_group': str(avg_age_group) if avg_age_group else None,
        'female_founder_pct': float(female_founder_pct),
        'patent_rate': float(patent_rate),
        'bootstrap_rate': float(bootstrap_rate),
        'top_sharks': {},
        'segments': {}
    }
    
    # Calculate top sharks for this industry
    for shark in sharks:
        inv_col = f'{shark} Investment Amount'
        if inv_col in industry_df.columns:
            investments = industry_df[inv_col].gt(0).sum()
            profile['top_sharks'][shark] = int(investments)
    
    profile['top_sharks'] = dict(sorted(
        profile['top_sharks'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:3])
    
    return profile

# Generate profiles for all major industries
top_industries = df['Industry'].value_counts().head(10).index.tolist()

print(f"\n🔍 Generating profiles for {len(top_industries)} industries...")

industry_profiles = {}
for industry in top_industries:
    profile = generate_industry_profile(industry)
    if profile:
        industry_profiles[industry] = profile
        print(f"   ✅ {industry}: {profile['total_pitches']} pitches, {profile['offer_rate']:.1%} offer rate")

# Save to JSON
with open('reports/industry_profiles.json', 'w') as f:
    json.dump(industry_profiles, f, indent=2)
print("\n✅ Saved: reports/industry_profiles.json")

# ============================================================================
# 7.2 INDUSTRY COMPARISON DASHBOARD
# ============================================================================
print("\n" + "="*80)
print("7.2 INDUSTRY COMPARISON DASHBOARD")
print("="*80)

# Create comparison DataFrame
comparison_data = []
for profile in industry_profiles.values():
    comparison_data.append({
        'Industry': profile['industry'],
        'Pitches': profile['total_pitches'],
        'Market Share': f"{profile['market_share']:.1f}%",
        'Offer Rate': f"{profile['offer_rate']:.1%}",
        'Avg Revenue': f"₹{profile['avg_revenue']:.0f}L",
        'Avg Deal': f"₹{profile['avg_deal_size']:.0f}L",
        'Avg Equity': f"{profile['avg_equity_given']:.1f}%",
        'Rev Multiple': f"{profile['revenue_multiple']:.1f}x",
        'Patent Rate': f"{profile['patent_rate']:.1f}%"
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n📊 Industry Comparison Table:")
print(comparison_df.to_string(index=False))

# Save comparison table
comparison_df.to_csv('reports/industry_comparison.csv', index=False)
print("\n✅ Saved: reports/industry_comparison.csv")

# Visualize key metrics
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

industries_list = [p['industry'] for p in industry_profiles.values()]
offer_rates = [p['offer_rate'] * 100 for p in industry_profiles.values()]
deal_sizes = [p['avg_deal_size'] for p in industry_profiles.values()]
revenue_multiples = [p['revenue_multiple'] for p in industry_profiles.values()]
equity_given = [p['avg_equity_given'] for p in industry_profiles.values()]

# Plot 1: Offer Rate by Industry
axes[0, 0].barh(industries_list, offer_rates, color='steelblue', alpha=0.8)
axes[0, 0].set_xlabel('Offer Rate (%)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Success Rate by Industry', fontsize=14, fontweight='bold', pad=15)
overall_avg = df['Received Offer'].mean() * 100
axes[0, 0].axvline(overall_avg, color='red', linestyle='--', linewidth=2, label=f'Overall Avg: {overall_avg:.1f}%')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Average Deal Size
axes[0, 1].barh(industries_list, deal_sizes, color='coral', alpha=0.8)
axes[0, 1].set_xlabel('Average Deal Size (₹ Lakhs)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Deal Size by Industry', fontsize=14, fontweight='bold', pad=15)
axes[0, 1].grid(axis='x', alpha=0.3)

# Plot 3: Revenue Multiple
axes[1, 0].barh(industries_list, revenue_multiples, color='mediumseagreen', alpha=0.8)
axes[1, 0].set_xlabel('Revenue Multiple (x)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Valuation Multiple by Industry', fontsize=14, fontweight='bold', pad=15)
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 4: Equity Given
axes[1, 1].barh(industries_list, equity_given, color='orchid', alpha=0.8)
axes[1, 1].set_xlabel('Average Equity Given (%)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Equity Dilution by Industry', fontsize=14, fontweight='bold', pad=15)
axes[1, 1].invert_xaxis()
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/industry_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: reports/figures/industry_comparison.png")
plt.close()

# ============================================================================
# 7.3 SUCCESS FACTORS ANALYSIS BY INDUSTRY
# ============================================================================
print("\n" + "="*80)
print("7.3 SUCCESS FACTORS ANALYSIS")
print("="*80)

def analyze_success_factors(industry_name):
    """Identify what drives success in each industry"""
    industry_df = df[df['Industry'] == industry_name].copy()
    
    if len(industry_df) < 10:
        return None
    
    # Split into successful and unsuccessful
    successful = industry_df[industry_df['Received Offer'] == 1]
    unsuccessful = industry_df[industry_df['Received Offer'] == 0]
    
    if len(successful) < 5 or len(unsuccessful) < 5:
        return None
    
    print(f"\n{'='*70}")
    print(f"🎯 SUCCESS FACTORS: {industry_name.upper()}")
    print(f"{'='*70}")
    print(f"Successful: {len(successful)} | Unsuccessful: {len(unsuccessful)}")
    
    # Compare key metrics
    comparisons = {
        'Avg Revenue': ('Yearly Revenue', 'median'),
        'Avg Gross Margin': ('Gross Margin', 'mean'),
        'Avg Net Margin': ('Net Margin', 'mean'),
        'Patent %': ('has_patent', 'mean'),
        'Bootstrapped %': ('is_bootstrapped', 'mean'),
        'Female Founder %': ('has_female_founder', 'mean'),
    }
    
    success_factors = {}
    
    print(f"\n📊 Key Metric Comparisons:")
    for metric_name, (column, agg_func) in comparisons.items():
        if column not in industry_df.columns:
            continue
        
        if agg_func == 'mean':
            success_val = successful[column].mean()
            fail_val = unsuccessful[column].mean()
        else:
            success_val = successful[column].median()
            fail_val = unsuccessful[column].median()
        
        if pd.notna(success_val) and pd.notna(fail_val) and fail_val != 0:
            diff_pct = ((success_val - fail_val) / fail_val) * 100
            success_factors[metric_name] = {
                'successful': float(success_val),
                'unsuccessful': float(fail_val),
                'difference_pct': float(diff_pct)
            }
            
            print(f"\n   {metric_name}:")
            print(f"      Successful: {success_val:.1f}")
            print(f"      Unsuccessful: {fail_val:.1f}")
            print(f"      Difference: {diff_pct:+.1f}%")
    
    # Top differentiating features (correlation analysis)
    print(f"\n🔍 Top Differentiators (Correlation with Success):")
    correlations = []
    
    numeric_cols = industry_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['Season Number', 'Episode Number', 'Pitch Number', 'Received Offer']:
            try:
                corr = industry_df[[col, 'Received Offer']].corr().iloc[0, 1]
                if abs(corr) > 0.15 and pd.notna(corr):
                    correlations.append((col, corr))
            except:
                pass
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in correlations[:5]:
        direction = "positive" if corr > 0 else "negative"
        print(f"   • {col}: {corr:+.3f} ({direction})")
    
    return success_factors

# Analyze all top industries
all_success_factors = {}
for industry in list(industry_profiles.keys())[:8]:  # Top 8 industries
    factors = analyze_success_factors(industry)
    if factors:
        all_success_factors[industry] = factors

# Save success factors
with open('reports/industry_success_factors.json', 'w') as f:
    json.dump(all_success_factors, f, indent=2)
print("\n✅ Saved: reports/industry_success_factors.json")

# ============================================================================
# 7.4 INDUSTRY TRENDS OVER SEASONS
# ============================================================================
print("\n" + "="*80)
print("7.4 INDUSTRY TRENDS OVER SEASONS")
print("="*80)

# Analyze how each industry evolved across seasons
industry_trends = df.groupby(['Season Number', 'Industry']).agg({
    'Received Offer': 'mean',
    'Total Deal Amount': 'mean',
    'Total Deal Equity': 'mean',
    'Valuation Requested': 'median'
}).reset_index()

industry_trends.columns = ['Season', 'Industry', 'Offer_Rate', 'Avg_Deal_Amount', 'Avg_Equity', 'Median_Valuation']

print("\n📈 Industry Trends Summary:")
print(industry_trends.head(20).to_string(index=False))

# Save trends
industry_trends.to_csv('reports/industry_trends.csv', index=False)
print("\n✅ Saved: reports/industry_trends.csv")

# Visualize trends for top 5 industries
top_5_industries = df['Industry'].value_counts().head(5).index

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Plot 1: Offer Rate Trends
for industry in top_5_industries:
    industry_data = industry_trends[industry_trends['Industry'] == industry]
    axes[0, 0].plot(industry_data['Season'], industry_data['Offer_Rate'] * 100, 
                    marker='o', linewidth=2, label=industry, markersize=8)

axes[0, 0].set_xlabel('Season', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Offer Rate (%)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Offer Rate Trends by Industry', fontsize=14, fontweight='bold', pad=15)
axes[0, 0].legend(fontsize=9, loc='best')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_xticks([1, 2, 3])

# Plot 2: Average Deal Amount Trends
for industry in top_5_industries:
    industry_data = industry_trends[industry_trends['Industry'] == industry]
    axes[0, 1].plot(industry_data['Season'], industry_data['Avg_Deal_Amount'], 
                    marker='s', linewidth=2, label=industry, markersize=8)

axes[0, 1].set_xlabel('Season', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Avg Deal Amount (₹L)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Deal Size Trends by Industry', fontsize=14, fontweight='bold', pad=15)
axes[0, 1].legend(fontsize=9, loc='best')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xticks([1, 2, 3])

# Plot 3: Equity Trends
for industry in top_5_industries:
    industry_data = industry_trends[industry_trends['Industry'] == industry]
    axes[1, 0].plot(industry_data['Season'], industry_data['Avg_Equity'], 
                    marker='^', linewidth=2, label=industry, markersize=8)

axes[1, 0].set_xlabel('Season', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Avg Equity Given (%)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Equity Dilution Trends by Industry', fontsize=14, fontweight='bold', pad=15)
axes[1, 0].legend(fontsize=9, loc='best')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xticks([1, 2, 3])

# Plot 4: Valuation Trends
for industry in top_5_industries:
    industry_data = industry_trends[industry_trends['Industry'] == industry]
    axes[1, 1].plot(industry_data['Season'], industry_data['Median_Valuation'], 
                    marker='D', linewidth=2, label=industry, markersize=8)

axes[1, 1].set_xlabel('Season', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Median Valuation (₹L)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Valuation Trends by Industry', fontsize=14, fontweight='bold', pad=15)
axes[1, 1].legend(fontsize=9, loc='best')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xticks([1, 2, 3])

plt.tight_layout()
plt.savefig('reports/figures/industry_trends.png', dpi=300, bbox_inches='tight')
print("✅ Saved: reports/figures/industry_trends.png")
plt.close()

# ============================================================================
# 7.5 INDUSTRY ENTRY REQUIREMENTS
# ============================================================================
print("\n" + "="*80)
print("7.5 INDUSTRY ENTRY REQUIREMENTS")
print("="*80)

def calculate_entry_requirements(industry_name):
    """Calculate minimum requirements to be competitive in an industry"""
    industry_df = df[df['Industry'] == industry_name].copy()
    successful = industry_df[industry_df['Received Offer'] == 1]
    
    if len(successful) < 5:
        return None
    
    requirements = {
        'industry': industry_name,
        'sample_size': int(len(successful)),
        'minimum_revenue': {
            '25th_percentile': float(successful['Yearly Revenue'].quantile(0.25)),
            '50th_percentile': float(successful['Yearly Revenue'].quantile(0.50)),
            'recommended': float(successful['Yearly Revenue'].quantile(0.33))
        },
        'minimum_gross_margin': {
            '25th_percentile': float(successful['Gross Margin'].quantile(0.25)) if 'Gross Margin' in successful.columns else 0,
            'recommended': float(successful['Gross Margin'].quantile(0.33)) if 'Gross Margin' in successful.columns else 0
        },
        'minimum_net_margin': {
            '25th_percentile': float(successful['Net Margin'].quantile(0.25)) if 'Net Margin' in successful.columns else 0,
            'recommended': float(successful['Net Margin'].quantile(0.33)) if 'Net Margin' in successful.columns else 0
        },
        'typical_ask_range': {
            'lower': float(successful['Original Ask Amount'].quantile(0.25)),
            'upper': float(successful['Original Ask Amount'].quantile(0.75)),
            'median': float(successful['Original Ask Amount'].median())
        },
        'typical_valuation_range': {
            'lower': float(successful['Valuation Requested'].quantile(0.25)),
            'upper': float(successful['Valuation Requested'].quantile(0.75)),
            'median': float(successful['Valuation Requested'].median())
        },
        'success_boosters': {}
    }
    
    # Success boosters
    if 'has_patent' in successful.columns:
        requirements['success_boosters']['has_patent'] = bool(successful['has_patent'].mean() > 0.3)
    if 'has_female_founder' in successful.columns:
        requirements['success_boosters']['female_founder'] = bool(successful['has_female_founder'].mean() > 0.35)
    if 'is_bootstrapped' in successful.columns:
        requirements['success_boosters']['bootstrapped'] = bool(successful['is_bootstrapped'].mean() > 0.25)
    
    return requirements

# Generate requirements for all industries
print("\n🎯 Calculating entry requirements for all industries...")

entry_requirements = {}
entry_requirements_list = []

for industry in industry_profiles.keys():
    req = calculate_entry_requirements(industry)
    if req:
        entry_requirements[industry] = req
        
        # Add to list for CSV
        entry_requirements_list.append({
            'Industry': industry,
            'Sample_Size': req['sample_size'],
            'Min_Revenue_Recommended': f"₹{req['minimum_revenue']['recommended']:.0f}L",
            'Min_Gross_Margin': f"{req['minimum_gross_margin']['recommended']:.1f}%",
            'Typical_Ask_Lower': f"₹{req['typical_ask_range']['lower']:.0f}L",
            'Typical_Ask_Upper': f"₹{req['typical_ask_range']['upper']:.0f}L",
            'Typical_Valuation_Median': f"₹{req['typical_valuation_range']['median']:.0f}L",
            'Patent_Booster': req['success_boosters'].get('has_patent', False),
            'Female_Founder_Booster': req['success_boosters'].get('female_founder', False),
            'Bootstrapped_Booster': req['success_boosters'].get('bootstrapped', False)
        })
        
        print(f"\n{'='*70}")
        print(f"📋 {industry}")
        print(f"{'='*70}")
        print(f"Min Revenue (to be competitive): ₹{req['minimum_revenue']['recommended']:.0f}L")
        print(f"Min Gross Margin: {req['minimum_gross_margin']['recommended']:.1f}%")
        print(f"Min Net Margin: {req['minimum_net_margin']['recommended']:.1f}%")
        print(f"Typical Ask Range: ₹{req['typical_ask_range']['lower']:.0f}L - ₹{req['typical_ask_range']['upper']:.0f}L")
        print(f"Typical Valuation: ₹{req['typical_valuation_range']['median']:.0f}L")
        boosters = [k for k, v in req['success_boosters'].items() if v]
        print(f"Success Boosters: {', '.join(boosters) if boosters else 'None identified'}")

# Save entry requirements
with open('reports/industry_entry_requirements.json', 'w') as f:
    json.dump(entry_requirements, f, indent=2)
print("\n✅ Saved: reports/industry_entry_requirements.json")

# Save as CSV
entry_req_df = pd.DataFrame(entry_requirements_list)
entry_req_df.to_csv('reports/industry_entry_requirements.csv', index=False)
print("✅ Saved: reports/industry_entry_requirements.csv")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("📊 INDUSTRY INTELLIGENCE SUMMARY")
print("="*80)

print(f"\n🏭 Industries Analyzed: {len(industry_profiles)}")
print(f"   Total Startups: {len(df)}")
print(f"   Total Deals: {df['got_offer'].sum()}")

print(f"\n🎯 Top 5 Industries by Volume:")
for i, (industry, profile) in enumerate(list(industry_profiles.items())[:5], 1):
    print(f"   {i}. {industry}: {profile['total_pitches']} pitches ({profile['market_share']:.1f}%)")

print(f"\n💰 Top 5 Industries by Success Rate:")
sorted_by_success = sorted(industry_profiles.items(), key=lambda x: x[1]['offer_rate'], reverse=True)
for i, (industry, profile) in enumerate(sorted_by_success[:5], 1):
    print(f"   {i}. {industry}: {profile['offer_rate']:.1%} offer rate")

print(f"\n📈 Top 5 Industries by Revenue Multiple:")
sorted_by_multiple = sorted(industry_profiles.items(), key=lambda x: x[1]['revenue_multiple'], reverse=True)
for i, (industry, profile) in enumerate(sorted_by_multiple[:5], 1):
    if profile['revenue_multiple'] > 0:
        print(f"   {i}. {industry}: {profile['revenue_multiple']:.1f}x")

print(f"\n💡 Key Insights:")
print(f"   • Most competitive industry: {sorted_by_success[0][0]} ({sorted_by_success[0][1]['offer_rate']:.1%} success)")
print(f"   • Highest valuation multiples: {sorted_by_multiple[0][0]} ({sorted_by_multiple[0][1]['revenue_multiple']:.1f}x)")
print(f"   • Largest market: {list(industry_profiles.items())[0][0]} ({list(industry_profiles.items())[0][1]['market_share']:.1f}%)")

print("\n" + "="*80)
print("✅ PHASE 7 ANALYSIS COMPLETE")
print("="*80)
print("\nDeliverables created:")
print("   ✅ Industry profiles (JSON)")
print("   ✅ Industry comparison dashboard (PNG)")
print("   ✅ Success factors analysis (JSON)")
print("   ✅ Industry trends (CSV + PNG)")
print("   ✅ Entry requirements (JSON + CSV)")
