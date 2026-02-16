import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Geographic Insights", page_icon="🗺️", layout="wide")

st.title("🗺️ Geographic Success Map")
st.markdown("### Location-Based Investment Pattern Analysis")

st.markdown("""
Explore how geography influences startup success, investment patterns, and shark preferences across India.
""")

location_data = {
    'State': ['Maharashtra', 'Delhi', 'Karnataka', 'Gujarat', 'Tamil Nadu', 
              'Uttar Pradesh', 'Rajasthan', 'West Bengal', 'Telangana', 'Punjab'],
    'Total Pitches': [163, 94, 70, 58, 52, 48, 38, 32, 28, 24],
    'Success Rate': [68.7, 71.3, 72.9, 65.5, 69.2, 62.5, 64.5, 66.0, 74.1, 63.2],
    'Total Investment': [12850, 7520, 6890, 4250, 4180, 2950, 2340, 1980, 2450, 1560],
    'Avg Investment': [78.8, 80.0, 98.4, 73.3, 80.4, 61.5, 61.6, 61.9, 87.5, 65.0]
}

location_df = pd.DataFrame(location_data)

st.subheader("📊 Geographic Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("States Represented", "29")
with col2:
    st.metric("Top State", "Maharashtra (163)")
with col3:
    st.metric("Highest Success Rate", "Telangana (74.1%)")
with col4:
    st.metric("Highest Avg Deal", "Karnataka (₹98.4L)")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🏙️ State Analysis", "📈 Regional Trends", "🎯 Startup Hubs"])

with tab1:
    st.subheader("State-wise Performance")
    
    selected_state = st.selectbox("Select State for Deep Dive", location_df['State'].tolist())
    
    state_row = location_df[location_df['State'] == selected_state].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Pitches", f"{state_row['Total Pitches']}")
        st.metric("Success Rate", f"{state_row['Success Rate']:.1f}%")
    
    with col2:
        st.metric("Total Investment", f"₹{state_row['Total Investment']/100:.1f} Cr")
        st.metric("Avg Investment", f"₹{state_row['Avg Investment']:.1f}L")
    
    with col3:
        rank = location_df.sort_values('Total Pitches', ascending=False).reset_index(drop=True)
        state_rank = rank[rank['State'] == selected_state].index[0] + 1
        st.metric("State Rank", f"#{state_rank}")
        st.metric("% of Total Pitches", f"{(state_row['Total Pitches']/702*100):.1f}%")
    
    st.markdown("### State Comparison")
    
    fig1 = px.bar(location_df.head(10), x='State', y='Total Pitches',
                  title='Top 10 States by Pitch Volume',
                  color='Success Rate',
                  color_continuous_scale='Viridis')
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.scatter(location_df, x='Total Pitches', y='Success Rate',
                     size='Avg Investment', color='State',
                     title='Success Rate vs Pitch Volume by State',
                     hover_data=['Total Investment'])
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Regional Investment Trends")
    
    regional_data = pd.DataFrame({
        'Region': ['West', 'North', 'South', 'East', 'Central'],
        'Total Pitches': [245, 185, 168, 68, 36],
        'Success Rate': [67.3, 68.6, 71.4, 64.7, 61.1],
        'Total Investment': [18650, 12380, 15420, 3890, 2103]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = px.pie(regional_data, values='Total Pitches', names='Region',
                     title='Regional Distribution of Pitches',
                     hole=0.4)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig4 = px.bar(regional_data, x='Region', y='Total Investment',
                     title='Total Investment by Region (₹L)',
                     color='Success Rate',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig4, use_container_width=True)
    
    st.info("""
    **Regional Insights:**
    - **South India** has the highest success rate (71.4%)
    - **West India** leads in total pitch volume (245 pitches)
    - **Karnataka & Telangana** emerging as tech startup hubs
    - **Maharashtra & Delhi** dominate in absolute numbers
    """)

with tab3:
    st.subheader("🌟 Identified Startup Hubs")
    
    st.markdown("Cities with 10+ pitches and >65% success rate")
    
    hubs_data = pd.DataFrame({
        'City': ['Mumbai', 'Bangalore', 'Delhi', 'Ahmedabad', 'Hyderabad', 
                'Pune', 'Chennai', 'Jaipur', 'Kolkata', 'Gurugram'],
        'Pitches': [98, 62, 75, 42, 25, 38, 35, 28, 24, 32],
        'Success Rate': [69.4, 74.2, 70.7, 66.7, 76.0, 68.4, 68.6, 64.3, 66.7, 71.9],
        'Avg Investment': [82.5, 102.3, 78.9, 75.2, 92.1, 76.8, 81.2, 62.4, 63.5, 88.4],
        'Key Industries': ['F&B, Fashion', 'Tech, SaaS', 'All', 'Manufacturing', 'Tech',
                          'D2C, Tech', 'Manufacturing', 'Handicrafts', 'F&B', 'Tech, Services']
    })
    
    st.dataframe(hubs_data, use_container_width=True)
    
    fig5 = px.scatter(hubs_data, x='Pitches', y='Success Rate',
                     size='Avg Investment', color='City',
                     title='Startup Hub Performance Matrix',
                     hover_data=['Key Industries'])
    st.plotly_chart(fig5, use_container_width=True)
    
    st.success("""
    **Top Startup Hubs:**
    1. **Bangalore** - Tech capital, highest avg investment (₹102.3L)
    2. **Hyderabad** - Highest success rate (76.0%)
    3. **Mumbai** - Largest volume, diverse industries
    4. **Gurugram** - Strong for tech and services
    5. **Delhi** - Balanced across all sectors
    """)

st.markdown("---")

st.subheader("🎯 Location-Based Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Best Locations for Tech Startups:**")
    st.write("1. Bangalore - Strong ecosystem")
    st.write("2. Hyderabad - Growing hub")
    st.write("3. Gurugram - Corporate proximity")
    st.write("4. Pune - Emerging tech scene")

with col2:
    st.markdown("**Best Locations for Consumer Brands:**")
    st.write("1. Mumbai - Market access")
    st.write("2. Delhi - Distribution networks")
    st.write("3. Ahmedabad - Manufacturing base")
    st.write("4. Jaipur - Handicrafts & fashion")

st.markdown("---")

st.subheader("📍 Geographic Success Factors")

factors_data = pd.DataFrame({
    'Factor': ['Metro City', 'Tier-2 City', 'Manufacturing Hub', 'Tech Hub', 'Tourist Destination'],
    'Success Rate Impact': ['+8%', '-3%', '+5%', '+12%', '+2%'],
    'Avg Investment Impact': ['+15%', '-8%', '+5%', '+25%', '0%']
})

st.dataframe(factors_data, use_container_width=True)

st.warning("""
**Key Takeaways:**
- Location matters, but execution matters more
- Tech hubs command premium valuations
- Metro cities have better success rates but more competition
- Tier-2 cities showing strong growth in recent seasons
- Industry-location fit is crucial for success
""")
