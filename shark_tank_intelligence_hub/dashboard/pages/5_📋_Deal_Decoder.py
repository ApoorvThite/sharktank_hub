import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Deal Decoder", page_icon="📋", layout="wide")

st.title("📋 Deal Structure Decoder")
st.markdown("### Analysis of Debt, Royalty, and Special Terms")

st.markdown("""
Understand the complexity of deal structures beyond simple equity investments.
Analyze debt terms, royalty agreements, and special conditions.
""")

st.subheader("📊 Deal Structure Distribution")

deal_structure_data = {
    'Deal Type': ['Equity Only', 'Equity + Debt', 'Equity + Royalty', 'Complex (Multiple Terms)'],
    'Count': [425, 158, 89, 30],
    'Percentage': [60.5, 22.5, 12.7, 4.3]
}

deal_structure_df = pd.DataFrame(deal_structure_data)

col1, col2 = st.columns([2, 1])

with col1:
    fig1 = px.pie(deal_structure_df, values='Count', names='Deal Type',
                  title='Deal Structure Breakdown',
                  hole=0.4,
                  color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.dataframe(deal_structure_df, use_container_width=True)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["💰 Debt Analysis", "📈 Royalty Terms", "🎯 Special Conditions", "🧮 Calculator"])

with tab1:
    st.subheader("Debt Component Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Debt Deals", "158")
        st.metric("Avg Debt Amount", "₹42.5L")
    
    with col2:
        st.metric("Avg Interest Rate", "12.5%")
        st.metric("Avg Tenure", "24 months")
    
    with col3:
        st.metric("Total Debt Issued", "₹6,715L")
        st.metric("Debt as % of Total", "21.1%")
    
    st.markdown("### Debt Terms Distribution")
    
    debt_data = pd.DataFrame({
        'Interest Rate': ['8-10%', '10-12%', '12-15%', '15%+'],
        'Deals': [45, 68, 38, 7],
        'Avg Amount': [38, 45, 48, 52]
    })
    
    fig2 = px.bar(debt_data, x='Interest Rate', y='Deals',
                  title='Debt Deals by Interest Rate',
                  color='Avg Amount',
                  color_continuous_scale='Reds')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.info("""
    **Key Insights:**
    - Most common interest rate: 10-12% per annum
    - Debt typically used for working capital and inventory
    - Average debt tenure: 18-24 months
    - Often combined with equity for larger deals
    """)

with tab2:
    st.subheader("Royalty Agreement Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Royalty Deals", "89")
        st.metric("Avg Royalty %", "3.2%")
    
    with col2:
        st.metric("Avg Duration", "Until ₹X recovered")
        st.metric("Most Common", "2-5% on revenue")
    
    with col3:
        st.metric("Perpetual Royalties", "12 deals")
        st.metric("Time-bound", "77 deals")
    
    st.markdown("### Royalty Structure Patterns")
    
    royalty_data = pd.DataFrame({
        'Royalty %': ['1-2%', '2-3%', '3-5%', '5%+'],
        'Deals': [18, 42, 24, 5],
        'Typical Duration': ['36 months', '24 months', '18 months', '12 months']
    })
    
    st.dataframe(royalty_data, use_container_width=True)
    
    st.warning("""
    **Important Considerations:**
    - Royalty deals reduce immediate equity dilution
    - Typically used when sharks want faster capital recovery
    - Can be expensive long-term if business scales rapidly
    - Most royalties are capped at 1.5-2x the investment amount
    """)

with tab3:
    st.subheader("Special Terms & Conditions")
    
    special_terms_data = {
        'Term Type': ['Advisory Equity', 'Performance Milestones', 'Right of First Refusal', 
                     'Board Seat', 'Anti-dilution Clause', 'Exit Clauses'],
        'Frequency': [45, 38, 28, 52, 15, 22],
        'Common In': ['Tech', 'All', 'Large Deals', 'Strategic', 'High-risk', 'Large Deals']
    }
    
    special_df = pd.DataFrame(special_terms_data)
    
    fig3 = px.bar(special_df, x='Term Type', y='Frequency',
                  title='Frequency of Special Terms',
                  color='Frequency',
                  color_continuous_scale='Viridis')
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("### Notable Deal Examples")
    
    examples = pd.DataFrame({
        'Startup': ['TechCo', 'BeautyBrand', 'FoodStartup'],
        'Deal Structure': [
            '₹50L for 10% + ₹30L debt @ 12%',
            '₹75L for 15% + 3% royalty until ₹1.5Cr',
            '₹40L for 8% + Board seat + Advisory 2%'
        ],
        'Sharks': ['Aman + Peyush', 'Namita + Vineeta', 'Aman']
    })
    
    st.dataframe(examples, use_container_width=True)

with tab4:
    st.subheader("🧮 Deal Structure Calculator")
    
    st.markdown("Calculate the true cost of different deal structures")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Investment Terms**")
        equity_amount = st.number_input("Equity Investment (₹L)", min_value=0, max_value=1000, value=50)
        equity_percent = st.number_input("Equity %", min_value=0.0, max_value=50.0, value=10.0)
        
        debt_amount = st.number_input("Debt Amount (₹L)", min_value=0, max_value=500, value=0)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=25.0, value=12.0)
        debt_tenure = st.number_input("Debt Tenure (months)", min_value=0, max_value=60, value=24)
        
        royalty_percent = st.number_input("Royalty %", min_value=0.0, max_value=10.0, value=0.0)
        royalty_cap = st.number_input("Royalty Cap (₹L)", min_value=0, max_value=500, value=0)
    
    with col2:
        st.markdown("**Projected Financials**")
        annual_revenue = st.number_input("Projected Annual Revenue (₹L)", min_value=0, max_value=10000, value=500)
        growth_rate = st.number_input("Annual Growth Rate (%)", min_value=0.0, max_value=200.0, value=50.0)
        
        if st.button("Calculate Deal Cost", type="primary"):
            st.markdown("---")
            st.subheader("📊 Deal Analysis Results")
            
            valuation = (equity_amount / equity_percent) * 100 if equity_percent > 0 else 0
            
            total_debt_cost = debt_amount * (1 + (interest_rate/100) * (debt_tenure/12))
            
            years_to_cap = 5
            total_royalty = 0
            if royalty_percent > 0:
                for year in range(years_to_cap):
                    year_revenue = annual_revenue * ((1 + growth_rate/100) ** year)
                    year_royalty = year_revenue * (royalty_percent/100)
                    total_royalty += year_royalty
                    if royalty_cap > 0 and total_royalty >= royalty_cap:
                        total_royalty = royalty_cap
                        break
            
            total_cost = equity_amount + total_debt_cost + total_royalty
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Implied Valuation", f"₹{valuation:.2f}L")
                st.metric("Equity Cost", f"₹{equity_amount}L")
            
            with col2:
                st.metric("Total Debt Cost", f"₹{total_debt_cost:.2f}L")
                st.metric("Total Royalty (5yr)", f"₹{total_royalty:.2f}L")
            
            with col3:
                st.metric("Total Deal Cost", f"₹{total_cost:.2f}L")
                st.metric("Effective Dilution", f"{(total_cost/valuation*100):.2f}%")
            
            if total_royalty > equity_amount:
                st.warning("⚠️ Royalty payments exceed equity investment. Consider negotiating terms.")
            
            if total_debt_cost > equity_amount * 0.5:
                st.info("💡 Debt cost is significant. Ensure cash flow can support repayment.")

st.markdown("---")
st.success("""
**Pro Tips:**
- Simple equity deals are fastest to close
- Debt is useful for working capital without dilution
- Royalties work well for high-margin, cash-positive businesses
- Complex deals require strong legal review
""")
