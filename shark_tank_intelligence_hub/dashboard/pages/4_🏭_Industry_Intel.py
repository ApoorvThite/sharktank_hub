import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Industry Intelligence", page_icon="🏭", layout="wide")

st.title("🏭 Industry Deep Dive Dashboard")
st.markdown("### Sector-Specific Intelligence & Benchmarks")

industries_data = {
    'Industry': ['Food & Beverage', 'Beauty & Fashion', 'Technology', 'Healthcare', 
                 'Education', 'Fitness', 'Home & Kitchen', 'Agriculture'],
    'Total Pitches': [154, 141, 98, 76, 54, 48, 42, 38],
    'Success Rate': [68.2, 72.3, 65.3, 70.1, 62.0, 58.3, 64.3, 60.5],
    'Avg Investment': [78.5, 85.2, 125.3, 95.4, 68.7, 62.3, 72.1, 58.9],
    'Total Investment': [8254, 8562, 9876, 5234, 2987, 2145, 2234, 1876]
}

industries_df = pd.DataFrame(industries_data)

st.subheader("📊 Industry Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Industries", "18")
with col2:
    st.metric("Most Pitched", "Food & Beverage")
with col3:
    st.metric("Highest Success Rate", "Beauty & Fashion (72.3%)")
with col4:
    st.metric("Highest Avg Deal", "Technology (₹125.3L)")

st.markdown("---")

selected_industry = st.selectbox("Select Industry for Deep Dive", industries_df['Industry'].tolist())

industry_row = industries_df[industries_df['Industry'] == selected_industry].iloc[0]

st.subheader(f"📈 {selected_industry} - Detailed Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Pitches", f"{industry_row['Total Pitches']}")
    st.metric("Success Rate", f"{industry_row['Success Rate']:.1f}%")

with col2:
    st.metric("Avg Investment", f"₹{industry_row['Avg Investment']:.1f}L")
    st.metric("Total Investment", f"₹{industry_row['Total Investment']/100:.1f} Cr")

with col3:
    rank = industries_df.sort_values('Total Pitches', ascending=False).reset_index(drop=True)
    industry_rank = rank[rank['Industry'] == selected_industry].index[0] + 1
    st.metric("Industry Rank", f"#{industry_rank}")
    st.metric("Market Share", f"{(industry_row['Total Pitches']/702*100):.1f}%")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Comparisons", "📈 Trends", "🦈 Shark Preferences"])

with tab1:
    st.subheader("Industry Comparison")
    
    fig1 = px.bar(industries_df, x='Industry', y='Total Pitches',
                  title='Total Pitches by Industry',
                  color='Total Pitches',
                  color_continuous_scale='Blues')
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.scatter(industries_df, x='Total Pitches', y='Success Rate',
                     size='Avg Investment', color='Industry',
                     title='Success Rate vs Pitch Volume',
                     hover_data=['Total Investment'])
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Seasonal Trends")
    
    trend_data = pd.DataFrame({
        'Season': [1, 2, 3, 4, 5],
        'Pitches': [28, 32, 35, 30, 29],
        'Success Rate': [65, 68, 72, 70, 71],
        'Avg Investment': [72, 75, 82, 85, 88]
    })
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=trend_data['Season'], y=trend_data['Pitches'],
                          name='Pitches', marker_color='lightblue'))
    fig3.add_trace(go.Scatter(x=trend_data['Season'], y=trend_data['Success Rate'],
                             name='Success Rate (%)', mode='lines+markers',
                             yaxis='y2', marker_color='red'))
    
    fig3.update_layout(
        title=f'{selected_industry} - Seasonal Trends',
        yaxis=dict(title='Number of Pitches'),
        yaxis2=dict(title='Success Rate (%)', overlaying='y', side='right'),
        hovermode='x'
    )
    
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Shark Investment Preferences")
    
    shark_pref_data = pd.DataFrame({
        'Shark': ['Aman', 'Namita', 'Peyush', 'Vineeta', 'Anupam'],
        'Investments': [42, 38, 35, 32, 28],
        'Avg Amount': [82, 78, 95, 75, 88]
    })
    
    fig4 = px.bar(shark_pref_data, x='Shark', y='Investments',
                  title=f'Shark Investments in {selected_industry}',
                  color='Avg Amount',
                  color_continuous_scale='Greens')
    st.plotly_chart(fig4, use_container_width=True)
    
    st.info(f"""
    **Top Sharks for {selected_industry}:**
    - Most Active: {shark_pref_data.iloc[0]['Shark']} ({shark_pref_data.iloc[0]['Investments']} deals)
    - Highest Avg Investment: {shark_pref_data.loc[shark_pref_data['Avg Amount'].idxmax(), 'Shark']} (₹{shark_pref_data['Avg Amount'].max():.0f}L)
    """)

st.markdown("---")

st.subheader("💡 Industry Insights")

insights = {
    'Food & Beverage': "High volume, moderate success. Focus on unique flavors and scalability.",
    'Beauty & Fashion': "Highest success rate. D2C brands with strong online presence perform well.",
    'Technology': "Highest average investment. B2B SaaS and AI/ML startups attract premium valuations.",
    'Healthcare': "Strong success rate. Regulatory compliance and clinical validation are key.",
    'Education': "Growing sector. EdTech with proven user engagement gets funded.",
    'Fitness': "Competitive space. Unique value proposition and retention metrics matter.",
    'Home & Kitchen': "Innovation-driven. Patents and design uniqueness increase chances.",
    'Agriculture': "Impact-focused. Scalability and farmer adoption are critical."
}

st.success(insights.get(selected_industry, "Industry-specific insights coming soon."))
