"""
PHASE 6: SHARK COLLABORATION NETWORK ANALYSIS
Complete network analysis with co-investment graphs, shark profiles, and recommendation engine
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import json
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("="*80)
print("🤝 PHASE 6: SHARK COLLABORATION NETWORK ANALYSIS")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading processed data...")
df = pd.read_csv('data/processed/processed_data_full.csv')

print(f"✅ Loaded {len(df)} startups")
print(f"   Deals: {df['got_offer'].sum()}")

# ============================================================================
# 6.1 NETWORK GRAPH CONSTRUCTION
# ============================================================================
print("\n" + "="*80)
print("6.1 CO-INVESTMENT NETWORK CONSTRUCTION")
print("="*80)

# Define sharks
sharks = ['Namita', 'Aman', 'Anupam', 'Peyush', 'Vineeta', 'Ritesh', 'Amit']

# Create co-investment matrix
co_investment_matrix = pd.DataFrame(0, index=sharks, columns=sharks)

print("\n🔍 Analyzing co-investment patterns...")

for idx, row in df[df['got_offer'] == 1].iterrows():
    invested_sharks = []
    for shark in sharks:
        col_name = f'{shark} Investment Amount'
        if col_name in df.columns and pd.notna(row[col_name]) and row[col_name] > 0:
            invested_sharks.append(shark)
    
    # Update matrix for all pairs
    for i in range(len(invested_sharks)):
        for j in range(i+1, len(invested_sharks)):
            shark1, shark2 = invested_sharks[i], invested_sharks[j]
            co_investment_matrix.loc[shark1, shark2] += 1
            co_investment_matrix.loc[shark2, shark1] += 1

print("\n📊 Co-Investment Matrix:")
print(co_investment_matrix)

# Save matrix
co_investment_matrix.to_csv('reports/co_investment_matrix.csv')
print("\n✅ Saved: reports/co_investment_matrix.csv")

# Top partnerships
print("\n🤝 Top Shark Partnerships:")
partnerships = []
for i in range(len(sharks)):
    for j in range(i+1, len(sharks)):
        count = co_investment_matrix.iloc[i, j]
        if count > 0:
            partnerships.append((sharks[i], sharks[j], count))

partnerships.sort(key=lambda x: x[2], reverse=True)
for i, (s1, s2, count) in enumerate(partnerships[:10], 1):
    print(f"   {i:2d}. {s1:10s} ↔ {s2:10s}: {count:3d} deals")

# ============================================================================
# 6.2 NETWORK VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("6.2 NETWORK GRAPH VISUALIZATION")
print("="*80)

# Create NetworkX graph
G = nx.Graph()

# Add nodes (sharks) with attributes
for shark in sharks:
    col_name = f'{shark} Investment Amount'
    if col_name in df.columns:
        total_deals = df[col_name].gt(0).sum()
        total_invested = df[col_name].sum()
    else:
        total_deals = 0
        total_invested = 0
    
    G.add_node(shark, deals=total_deals, invested=total_invested)

# Add edges (partnerships)
for i in range(len(sharks)):
    for j in range(i+1, len(sharks)):
        weight = co_investment_matrix.iloc[i, j]
        if weight > 0:
            G.add_edge(sharks[i], sharks[j], weight=weight)

# Calculate network metrics
density = nx.density(G)
avg_clustering = nx.average_clustering(G)

print(f"\n📊 Network Metrics:")
print(f"   Nodes (Sharks): {G.number_of_nodes()}")
print(f"   Edges (Partnerships): {G.number_of_edges()}")
print(f"   Network Density: {density:.3f}")
print(f"   Average Clustering: {avg_clustering:.3f}")

# Centrality measures
degree_cent = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G)
closeness_cent = nx.closeness_centrality(G)

print(f"\n🎯 Degree Centrality (Most Connected):")
for shark, score in sorted(degree_cent.items(), key=lambda x: x[1], reverse=True):
    print(f"   {shark:10s}: {score:.3f}")

print(f"\n🌉 Betweenness Centrality (Bridge Sharks):")
for shark, score in sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True):
    print(f"   {shark:10s}: {score:.3f}")

# Visualize network
fig, ax = plt.subplots(figsize=(16, 12))
pos = nx.spring_layout(G, k=2, seed=42, iterations=50)

# Node sizes based on total investments
node_sizes = [G.nodes[shark]['invested']/5 for shark in sharks]

# Edge widths based on co-investment count
edge_widths = [G[u][v]['weight']/3 for u, v in G.edges()]

# Draw network
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                       alpha=0.9, edgecolors='navy', linewidths=2, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', ax=ax)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray', ax=ax)

# Add edge labels
edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9, ax=ax)

ax.set_title("Shark Tank India: Co-Investment Network\n(Node size = Total invested, Edge width = Co-investments)", 
             fontsize=18, fontweight='bold', pad=20)
ax.axis('off')
plt.tight_layout()
plt.savefig('reports/figures/shark_network.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: reports/figures/shark_network.png")
plt.close()

# ============================================================================
# 6.3 COMMUNITY DETECTION
# ============================================================================
print("\n" + "="*80)
print("6.3 COMMUNITY DETECTION")
print("="*80)

# Simple community detection using greedy modularity
communities_dict = nx.community.greedy_modularity_communities(G)
communities = {}
for i, community in enumerate(communities_dict):
    for shark in community:
        communities[shark] = i

print(f"\n🔍 Detected {len(communities_dict)} Communities:")
for i, community in enumerate(communities_dict):
    sharks_in_community = list(community)
    print(f"   Community {i+1}: {', '.join(sharks_in_community)}")

# Visualize communities
fig, ax = plt.subplots(figsize=(16, 12))
pos = nx.spring_layout(G, k=2, seed=42, iterations=50)

# Color nodes by community
community_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#95E1D3']
node_colors = [community_colors[communities[shark]] for shark in sharks]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                       alpha=0.9, edgecolors='navy', linewidths=2, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', ax=ax)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, edge_color='gray', ax=ax)

ax.set_title("Shark Communities Based on Co-Investment Patterns", 
             fontsize=18, fontweight='bold', pad=20)
ax.axis('off')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=community_colors[i], label=f'Community {i+1}') 
                   for i in range(len(communities_dict))]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('reports/figures/shark_communities.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: reports/figures/shark_communities.png")
plt.close()

# ============================================================================
# 6.4 SHARK PERSONALITY PROFILES
# ============================================================================
print("\n" + "="*80)
print("6.4 SHARK PERSONALITY PROFILES")
print("="*80)

def create_shark_profile(shark_name):
    """Generate comprehensive profile for each shark"""
    shark = shark_name
    
    # Column names
    inv_col = f'{shark} Investment Amount'
    eq_col = f'{shark} Investment Equity'
    present_col = f'{shark} Present'
    
    # Check if columns exist
    if inv_col not in df.columns:
        return None
    
    # Investment statistics
    total_deals = df[inv_col].gt(0).sum()
    total_invested = df[inv_col].sum()
    avg_deal_size = total_invested / total_deals if total_deals > 0 else 0
    
    # Deals where shark was present
    present = df[present_col].sum() if present_col in df.columns else 0
    investment_rate = total_deals / present if present > 0 else 0
    
    # Average equity
    deals_df = df[df[inv_col] > 0].copy()
    avg_equity = deals_df[eq_col].mean() if eq_col in df.columns and len(deals_df) > 0 else 0
    
    # Top industries
    top_industries = deals_df['Industry'].value_counts().head(5).to_dict() if len(deals_df) > 0 else {}
    
    # Preferred partners
    partner_counts = {}
    for other_shark in sharks:
        if other_shark != shark:
            count = co_investment_matrix.loc[shark, other_shark]
            if count > 0:
                partner_counts[other_shark] = int(count)
    top_partners = sorted(partner_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Success rate of deals
    success_rate = deals_df['Accepted Offer'].mean() if len(deals_df) > 0 else 0
    
    # Revenue preference
    avg_revenue = deals_df['Yearly Revenue'].mean() if len(deals_df) > 0 else 0
    
    # Valuation preference
    avg_valuation = deals_df['Valuation Requested'].mean() if len(deals_df) > 0 else 0
    
    profile = {
        'name': shark,
        'total_deals': int(total_deals),
        'total_invested': float(total_invested),
        'avg_deal_size': float(avg_deal_size),
        'avg_equity': float(avg_equity),
        'episodes_present': int(present),
        'investment_rate': float(investment_rate),
        'top_industries': top_industries,
        'top_partners': top_partners,
        'deal_success_rate': float(success_rate),
        'avg_startup_revenue': float(avg_revenue),
        'avg_startup_valuation': float(avg_valuation),
        'community': int(communities.get(shark, 0))
    }
    
    return profile

# Create profiles for all sharks
shark_profiles = {}
for shark in sharks:
    profile = create_shark_profile(shark)
    if profile:
        shark_profiles[shark] = profile

# Print profiles
for shark, profile in shark_profiles.items():
    print(f"\n{'='*70}")
    print(f"🦈 {shark.upper()} PROFILE")
    print(f"{'='*70}")
    print(f"Total Investments: {profile['total_deals']} deals")
    print(f"Total Amount: ₹{profile['total_invested']:.0f}L")
    print(f"Average Deal: ₹{profile['avg_deal_size']:.1f}L for {profile['avg_equity']:.1f}% equity")
    print(f"Episodes Present: {profile['episodes_present']}")
    print(f"Investment Rate: {profile['investment_rate']:.1%}")
    print(f"Deal Success Rate: {profile['deal_success_rate']:.1%}")
    print(f"Avg Startup Revenue: ₹{profile['avg_startup_revenue']:.0f}L")
    print(f"Avg Startup Valuation: ₹{profile['avg_startup_valuation']:.0f}L")
    
    print(f"\n📊 Top Industries:")
    for ind, count in list(profile['top_industries'].items())[:3]:
        pct = count / profile['total_deals'] * 100 if profile['total_deals'] > 0 else 0
        print(f"   {ind}: {count} deals ({pct:.1f}%)")
    
    print(f"\n🤝 Preferred Partners:")
    for partner, count in profile['top_partners']:
        print(f"   {partner}: {count} co-investments")

# Save profiles
with open('reports/shark_profiles.json', 'w') as f:
    json.dump(shark_profiles, f, indent=2)
print("\n✅ Saved: reports/shark_profiles.json")

# ============================================================================
# 6.5 INDUSTRY-SHARK AFFINITY MATRIX
# ============================================================================
print("\n" + "="*80)
print("6.5 SHARK-INDUSTRY AFFINITY ANALYSIS")
print("="*80)

# Get top industries
top_industries = df['Industry'].value_counts().head(15).index.tolist()

# Create affinity matrix
affinity_matrix = pd.DataFrame(0.0, index=top_industries, columns=sharks)

for shark in sharks:
    inv_col = f'{shark} Investment Amount'
    if inv_col not in df.columns:
        continue
    
    total_shark_investments = df[inv_col].gt(0).sum()
    
    if total_shark_investments > 0:
        for industry in top_industries:
            industry_deals = df[df['Industry'] == industry]
            shark_investments = industry_deals[inv_col].gt(0).sum()
            affinity_matrix.loc[industry, shark] = (shark_investments / total_shark_investments * 100)

print("\n📊 Shark-Industry Affinity Matrix (% of portfolio):")
print(affinity_matrix.round(1))

# Save affinity matrix
affinity_matrix.to_csv('reports/shark_industry_affinity.csv')
print("\n✅ Saved: reports/shark_industry_affinity.csv")

# Visualize affinity heatmap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(affinity_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': '% of Shark Portfolio'}, ax=ax,
            linewidths=0.5, linecolor='gray')
ax.set_title('Shark-Industry Affinity Heatmap\n(Percentage of each shark\'s portfolio)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Shark', fontsize=13, fontweight='bold')
ax.set_ylabel('Industry', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('reports/figures/shark_industry_affinity.png', dpi=300, bbox_inches='tight')
print("✅ Saved: reports/figures/shark_industry_affinity.png")
plt.close()

# Top specializations
print("\n🎯 Shark Specializations (Top Industry %):")
for shark in sharks:
    if shark in affinity_matrix.columns:
        top_ind = affinity_matrix[shark].idxmax()
        top_pct = affinity_matrix[shark].max()
        print(f"   {shark:10s}: {top_ind:30s} ({top_pct:.1f}%)")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("📊 NETWORK ANALYSIS SUMMARY")
print("="*80)

print(f"\n🤝 Network Statistics:")
print(f"   Total Sharks: {len(sharks)}")
print(f"   Total Partnerships: {G.number_of_edges()}")
print(f"   Network Density: {density:.3f}")
print(f"   Average Clustering: {avg_clustering:.3f}")
print(f"   Communities Detected: {len(communities_dict)}")

print(f"\n💰 Investment Statistics:")
total_all_investments = sum(profile['total_invested'] for profile in shark_profiles.values())
total_all_deals = sum(profile['total_deals'] for profile in shark_profiles.values())
print(f"   Total Amount Invested: ₹{total_all_investments:.0f}L")
print(f"   Total Deals: {total_all_deals}")
print(f"   Average Deal Size: ₹{total_all_investments/total_all_deals:.1f}L")

print(f"\n🦈 Most Active Sharks:")
sorted_sharks = sorted(shark_profiles.items(), key=lambda x: x[1]['total_deals'], reverse=True)
for i, (shark, profile) in enumerate(sorted_sharks[:5], 1):
    print(f"   {i}. {shark}: {profile['total_deals']} deals, ₹{profile['total_invested']:.0f}L")

print(f"\n🤝 Strongest Partnerships:")
for i, (s1, s2, count) in enumerate(partnerships[:5], 1):
    print(f"   {i}. {s1} ↔ {s2}: {count} deals")

print("\n" + "="*80)
print("✅ PHASE 6.1-6.5 COMPLETE")
print("="*80)
print("\nNext: Building Shark Recommendation Engine (6.6)")
