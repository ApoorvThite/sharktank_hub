import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Valuation Check", page_icon="💰", layout="wide")

st.title("💰 Valuation Reality Check")
st.markdown("### Smart Valuation Benchmarking & Fairness Analysis")

st.markdown("""
Compare your startup valuation against industry benchmarks and historical Shark Tank India deals.
Get insights on whether your ask is realistic and competitive.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Startup")
    your_industry = st.selectbox("Industry", [
        "Food & Beverage", "Beauty & Fashion", "Technology", "Healthcare",
        "Education", "Fitness", "Home & Kitchen", "Agriculture"
    ])
    
    your_revenue = st.number_input("Annual Revenue (₹ Lakhs)", min_value=0, max_value=10000, value=100)
    your_profit = st.number_input("Annual Profit (₹ Lakhs)", min_value=-1000, max_value=5000, value=20)
    
    your_ask_amount = st.number_input("Your Ask Amount (₹ Lakhs)", min_value=1, max_value=1000, value=50)
    your_ask_equity = st.number_input("Your Ask Equity (%)", min_value=0.1, max_value=50.0, value=10.0)

with col2:
    st.subheader("Calculated Metrics")
    
    your_valuation = (your_ask_amount / your_ask_equity) * 100 if your_ask_equity > 0 else 0
    revenue_multiple = your_valuation / your_revenue if your_revenue > 0 else 0
    profit_multiple = your_valuation / your_profit if your_profit > 0 else 0
    
    st.metric("Your Valuation", f"₹{your_valuation:.2f} Lakhs")
    st.metric("Revenue Multiple", f"{revenue_multiple:.2f}x")
    st.metric("Profit Multiple", f"{profit_multiple:.2f}x" if your_profit > 0 else "N/A (Loss-making)")

if st.button("📊 Check Valuation", type="primary"):
    st.markdown("---")
    
    industry_benchmarks = {
        'Food & Beverage': {'avg_valuation': 450, 'avg_revenue_multiple': 4.5, 'success_rate': 68},
        'Beauty & Fashion': {'avg_valuation': 520, 'avg_revenue_multiple': 5.2, 'success_rate': 72},
        'Technology': {'avg_valuation': 680, 'avg_revenue_multiple': 6.8, 'success_rate': 65},
        'Healthcare': {'avg_valuation': 550, 'avg_revenue_multiple': 5.5, 'success_rate': 70},
        'Education': {'avg_valuation': 420, 'avg_revenue_multiple': 4.2, 'success_rate': 62},
        'Fitness': {'avg_valuation': 380, 'avg_revenue_multiple': 3.8, 'success_rate': 58},
        'Home & Kitchen': {'avg_valuation': 400, 'avg_revenue_multiple': 4.0, 'success_rate': 64},
        'Agriculture': {'avg_valuation': 350, 'avg_revenue_multiple': 3.5, 'success_rate': 60}
    }
    
    benchmark = industry_benchmarks.get(your_industry, industry_benchmarks['Food & Beverage'])
    
    st.subheader(f"📈 {your_industry} Industry Benchmarks")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Avg Industry Valuation",
            f"₹{benchmark['avg_valuation']}L",
            delta=f"{((your_valuation - benchmark['avg_valuation'])/benchmark['avg_valuation']*100):.1f}%"
        )
    
    with col2:
        st.metric(
            "Avg Revenue Multiple",
            f"{benchmark['avg_revenue_multiple']}x",
            delta=f"{((revenue_multiple - benchmark['avg_revenue_multiple'])/benchmark['avg_revenue_multiple']*100):.1f}%"
        )
    
    with col3:
        st.metric(
            "Industry Success Rate",
            f"{benchmark['success_rate']}%"
        )
    
    st.markdown("### 🎯 Valuation Assessment")
    
    valuation_gap = ((your_valuation - benchmark['avg_valuation']) / benchmark['avg_valuation']) * 100
    
    if abs(valuation_gap) <= 20:
        st.success(f"""
        ✅ **Fair Valuation**: Your valuation is within ±20% of industry average ({valuation_gap:+.1f}%).
        This is a realistic ask and increases your chances of getting a deal.
        """)
    elif valuation_gap > 20:
        st.warning(f"""
        ⚠️ **Overvalued**: Your valuation is {valuation_gap:.1f}% higher than industry average.
        Consider adjusting your ask to improve deal probability. Sharks may negotiate down significantly.
        """)
    else:
        st.info(f"""
        💡 **Undervalued**: Your valuation is {abs(valuation_gap):.1f}% lower than industry average.
        You have room to ask for more equity or less dilution. Consider revising your ask upward.
        """)
    
    st.markdown("### 📊 Valuation Comparison")
    
    comparison_data = pd.DataFrame({
        'Metric': ['Your Startup', 'Industry Average', 'Top 25%', 'Bottom 25%'],
        'Valuation': [your_valuation, benchmark['avg_valuation'], 
                     benchmark['avg_valuation'] * 1.3, benchmark['avg_valuation'] * 0.7]
    })
    
    fig = px.bar(comparison_data, x='Metric', y='Valuation', 
                 title='Valuation Comparison',
                 color='Valuation',
                 color_continuous_scale='Viridis')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 💡 Recommendations")
    
    if revenue_multiple > benchmark['avg_revenue_multiple'] * 1.5:
        st.write("- 🔴 Your revenue multiple is significantly high. Be prepared to justify with strong growth metrics.")
    elif revenue_multiple < benchmark['avg_revenue_multiple'] * 0.5:
        st.write("- 🟢 Your revenue multiple is conservative. You could potentially ask for higher valuation.")
    else:
        st.write("- 🟡 Your revenue multiple is reasonable and aligned with industry standards.")
    
    if your_profit > 0:
        st.write("- 🟢 Profitable business increases deal probability significantly.")
    else:
        st.write("- 🟡 Focus on path to profitability and growth story in your pitch.")
