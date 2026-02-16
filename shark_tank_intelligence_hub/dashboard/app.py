import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Shark Tank India Intelligence Hub",
    page_icon="🦈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🦈 Shark Tank India Intelligence Hub")
st.markdown("### Complete Data Science & Business Analytics Platform")

st.markdown("""
Welcome to the **Shark Tank India Intelligence Hub** - a comprehensive analytical platform 
analyzing 702 pitches from Seasons 1-5 (Dec 2021 - Feb 2026).

#### 📊 Key Statistics
- **Total Pitches**: 702 startups across 5 seasons
- **Success Rate**: 66.2% received offers, 86.2% accepted
- **Total Investment**: ₹318.4 Crores
- **Average Deal**: ₹79.4 Lakhs for 7.5% equity
- **Industries**: 18 categories
- **Geographic Spread**: 29 states

#### 🎯 Platform Modules

Navigate using the sidebar to access:

1. **🎯 Shark Predictor** - ML-powered deal prediction and shark selection
2. **💰 Valuation Check** - Smart valuation benchmarking and fairness analysis
3. **🤝 Shark Networks** - Network analysis of shark partnerships
4. **🏭 Industry Intel** - Sector-specific intelligence and benchmarks
5. **📋 Deal Decoder** - Analysis of debt, royalty, and special terms
6. **🗺️ Geo Insights** - Location-based investment patterns

#### 🚀 Getting Started

1. Upload your Shark Tank India dataset in the sidebar
2. Explore different analytical modules
3. Get insights, predictions, and benchmarks
4. Download reports and visualizations

---
*Built with Python, Streamlit, XGBoost, NetworkX, and Plotly*
""")

st.sidebar.title("Navigation")
st.sidebar.info("Use the pages above to navigate through different modules")

st.sidebar.title("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Shark Tank Dataset (CSV)", type=['csv'])

if uploaded_file is not None:
    st.sidebar.success("✅ Dataset loaded successfully!")
    st.sidebar.info(f"File: {uploaded_file.name}")
else:
    st.sidebar.warning("⚠️ Please upload the dataset to begin analysis")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This platform provides end-to-end data science capabilities including:
- Exploratory Data Analysis
- Machine Learning Predictions
- Network Analysis
- Business Intelligence
- Interactive Visualizations
""")
