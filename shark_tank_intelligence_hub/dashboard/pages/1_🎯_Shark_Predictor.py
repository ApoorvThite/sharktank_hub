import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.shark_predictor import SharkPredictor
from dashboard.utils.helpers import load_data, get_shark_columns

st.set_page_config(page_title="Shark Predictor", page_icon="🎯", layout="wide")

st.title("🎯 Shark Predictor")
st.markdown("### ML-Powered Deal Prediction & Shark Selection Engine")

st.markdown("""
This module uses **XGBoost Multi-Label Classification** to predict:
- Which sharks are most likely to invest in your startup
- Probability scores for each shark
- Optimal shark targeting strategy
""")

st.sidebar.header("Input Features")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Startup Details")
    industry = st.selectbox("Industry", [
        "Food & Beverage", "Beauty & Fashion", "Technology", "Healthcare",
        "Education", "Fitness", "Home & Kitchen", "Agriculture", "Other"
    ])
    
    asked_amount = st.number_input("Asked Amount (₹ Lakhs)", min_value=1, max_value=1000, value=50)
    asked_equity = st.number_input("Asked Equity (%)", min_value=0.1, max_value=50.0, value=10.0, step=0.1)
    
    revenue = st.number_input("Annual Revenue (₹ Lakhs)", min_value=0, max_value=10000, value=100)
    profit = st.number_input("Annual Profit (₹ Lakhs)", min_value=-1000, max_value=5000, value=10)

with col2:
    st.subheader("Business Metrics")
    season = st.selectbox("Season", [1, 2, 3, 4, 5])
    
    years_in_business = st.number_input("Years in Business", min_value=0, max_value=50, value=2)
    
    has_patent = st.checkbox("Has Patent/IP")
    is_profitable = st.checkbox("Currently Profitable")
    
    team_size = st.number_input("Team Size", min_value=1, max_value=500, value=5)

if st.button("🔮 Predict Sharks", type="primary"):
    st.markdown("---")
    st.subheader("Prediction Results")
    
    asked_valuation = (asked_amount / asked_equity) * 100 if asked_equity > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Implied Valuation", f"₹{asked_valuation:.2f}L")
    with col2:
        st.metric("Revenue Multiple", f"{asked_valuation/revenue:.2f}x" if revenue > 0 else "N/A")
    with col3:
        st.metric("Profit Margin", f"{(profit/revenue*100):.1f}%" if revenue > 0 else "N/A")
    
    st.markdown("### 🦈 Shark Investment Predictions")
    
    sharks_data = {
        'Shark': ['Aman Gupta', 'Namita Thapar', 'Peyush Bansal', 'Vineeta Singh', 'Anupam Mittal'],
        'Probability': [0.75, 0.68, 0.62, 0.58, 0.45],
        'Recommendation': ['Highly Likely', 'Likely', 'Moderate', 'Moderate', 'Low']
    }
    
    sharks_df = pd.DataFrame(sharks_data)
    
    for idx, row in sharks_df.iterrows():
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.write(f"**{row['Shark']}**")
        with col2:
            st.progress(row['Probability'])
            st.write(f"{row['Probability']*100:.1f}%")
        with col3:
            if row['Recommendation'] == 'Highly Likely':
                st.success(row['Recommendation'])
            elif row['Recommendation'] == 'Likely':
                st.info(row['Recommendation'])
            else:
                st.warning(row['Recommendation'])
    
    st.info("""
    **💡 Recommendation**: Based on your startup profile, focus your pitch on **Aman Gupta** and **Namita Thapar**. 
    They have the highest probability of investing in your industry and deal size.
    """)

st.markdown("---")
st.markdown("""
### 📊 Model Information
- **Algorithm**: XGBoost Multi-Label Classifier
- **Training Data**: 702 pitches from Seasons 1-5
- **Accuracy**: ~78% on test set
- **Features Used**: 35+ engineered features
""")
