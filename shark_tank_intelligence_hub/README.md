# 🦈 Shark Tank India Intelligence Hub

> **Complete Data Science & Business Analytics Platform**  
> Analyzing 702 pitches from Shark Tank India Seasons 1-5 (Dec 2021 - Feb 2026)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📊 Executive Summary

This comprehensive project combines **7 major analytical modules** into one unified intelligence platform, demonstrating end-to-end data science capabilities including exploratory data analysis, machine learning, network analysis, business intelligence, and interactive visualization.

### Key Dataset Statistics

- **Total Pitches**: 702 startups across 5 seasons
- **Success Metrics**: 66.2% received offers, 86.2% accepted when offered
- **Total Investment**: ₹31,843 Lakhs (~₹318.4 Crores)
- **Average Deal**: ₹79.4 Lakhs for 7.5% equity
- **Industries**: 18 categories led by Food & Beverage (154) and Beauty/Fashion (141)
- **Geographic Spread**: 29 states with Maharashtra (163), Delhi (94), Karnataka (70) leading
- **Sharks**: 7 main sharks + guests across all episodes

---

## 🎯 Project Objectives

This integrated project delivers **7 interconnected analytical modules**:

1. **📊 Exploratory Data Analysis (EDA)** - Comprehensive data understanding and pattern discovery
2. **🎯 The Shark Predictor** - ML-powered deal prediction and shark selection engine
3. **💰 Valuation Reality Check** - Smart valuation benchmarking and fairness analysis
4. **🤝 Shark Collaboration Network** - Network analysis of shark partnerships and synergies
5. **🏭 Industry Deep Dive Dashboard** - Sector-specific intelligence and benchmarks
6. **📋 Deal Structure Decoder** - Analysis of debt, royalty, and special terms
7. **🗺️ Geographic Success Map** - Location-based investment pattern analysis

### Value Proposition

- **For Founders**: Optimize pitch strategy, realistic valuation, shark targeting
- **For Investors**: Pattern recognition, due diligence insights, market intelligence
- **For Researchers**: Understanding Indian startup ecosystem dynamics
- **For Data Science Portfolio**: Showcase end-to-end ML + BA capabilities

---

## 📁 Project Structure

```
shark_tank_intelligence_hub/
│
├── data/
│   ├── raw/                          # Original CSV (702 rows × 80 cols)
│   ├── processed/                    # Cleaned data with engineered features
│   ├── industry_benchmarks/          # Industry-specific metrics
│   └── shark_profiles/               # Individual shark statistics
│
├── notebooks/
│   ├── 01_eda_comprehensive.ipynb        # MODULE 1: Deep EDA
│   ├── 02_feature_engineering.ipynb      # Data prep for ML
│   ├── 03_shark_predictor.ipynb          # MODULE 2: ML models
│   ├── 04_valuation_analysis.ipynb       # MODULE 3: Valuation
│   ├── 05_network_analysis.ipynb         # MODULE 4: Shark networks
│   ├── 06_industry_intelligence.ipynb    # MODULE 5: Industry insights
│   ├── 07_deal_structure.ipynb           # MODULE 6: Deal terms
│   └── 08_geographic_analysis.ipynb      # MODULE 7: Location patterns
│
├── src/
│   ├── data/
│   │   ├── loader.py                     # Data loading utilities
│   │   ├── cleaner.py                    # Data cleaning pipeline
│   │   └── feature_engineer.py           # Feature creation (35+ features)
│   │
│   ├── models/
│   │   ├── shark_predictor.py            # XGBoost multi-label classifier
│   │   ├── valuation_model.py            # Regression for equity dilution
│   │   └── model_explainer.py            # SHAP analysis
│   │
│   ├── analysis/
│   │   ├── network_analyzer.py           # NetworkX graph analysis
│   │   ├── industry_profiler.py          # Industry-specific metrics
│   │   ├── deal_decoder.py               # Deal structure patterns
│   │   └── geo_mapper.py                 # Geographic analysis
│   │
│   └── visualization/
│       ├── plotly_interactive.py         # Interactive charts
│       ├── network_viz.py                # Shark collaboration graphs
│       └── dashboards.py                 # Streamlit components
│
├── models/
│   ├── shark_predictor_xgb.pkl           # Trained XGBoost model
│   ├── equity_predictor_rf.pkl           # Random Forest regressor
│   └── feature_scaler.pkl                # StandardScaler object
│
├── dashboard/
│   ├── app.py                            # Main Streamlit app
│   ├── pages/
│   │   ├── 1_🎯_Shark_Predictor.py      # ML prediction interface
│   │   ├── 2_💰_Valuation_Check.py      # Valuation calculator
│   │   ├── 3_🤝_Shark_Networks.py       # Network visualization
│   │   ├── 4_🏭_Industry_Intel.py       # Industry benchmarks
│   │   ├── 5_📋_Deal_Decoder.py         # Deal structure analysis
│   │   └── 6_🗺️_Geo_Insights.py        # Geographic patterns
│   └── utils/
│       └── helpers.py                    # Shared functions
│
├── reports/
│   ├── figures/                          # All generated visualizations
│   ├── comprehensive_report.pdf          # Full technical report
│   └── executive_summary.pptx            # Business presentation
│
├── requirements.txt                      # Python dependencies
├── README.md                             # Project documentation
└── config.yaml                           # Configuration file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd shark_tank_intelligence_hub
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place your dataset**
   - Add `shark_tank_india.csv` to `data/raw/` directory

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Running Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and open any analysis notebook.

---

## 📊 Modules Overview

### 1. 🎯 Shark Predictor
**ML-Powered Deal Prediction Engine**

- **Algorithm**: XGBoost Multi-Label Classifier
- **Features**: 35+ engineered features
- **Accuracy**: ~78% on test set
- **Output**: Probability scores for each shark, optimal targeting strategy

**Use Case**: Input your startup details and get predictions on which sharks are most likely to invest.

---

### 2. 💰 Valuation Reality Check
**Smart Valuation Benchmarking**

- **Analysis**: Compare your valuation against industry benchmarks
- **Metrics**: Revenue multiples, profit multiples, valuation gaps
- **Recommendations**: Fair/Overvalued/Undervalued assessment

**Use Case**: Validate if your ask is realistic and competitive.

---

### 3. 🤝 Shark Collaboration Network
**Network Analysis of Partnerships**

- **Technology**: NetworkX graph analysis
- **Metrics**: Degree centrality, betweenness, influence scores
- **Visualization**: Interactive network graphs

**Use Case**: Understand shark partnership patterns and synergies.

---

### 4. 🏭 Industry Deep Dive
**Sector-Specific Intelligence**

- **Coverage**: 18 industry categories
- **Metrics**: Success rates, avg investments, trends
- **Benchmarks**: Industry-specific performance indicators

**Use Case**: Get industry-specific insights and benchmarks.

---

### 5. 📋 Deal Structure Decoder
**Analysis of Complex Deal Terms**

- **Components**: Equity, debt, royalty, special terms
- **Calculator**: Effective cost calculator
- **Patterns**: Common deal structures

**Use Case**: Understand and calculate true cost of different deal structures.

---

### 6. 🗺️ Geographic Success Map
**Location-Based Investment Patterns**

- **Coverage**: 29 states, 100+ cities
- **Analysis**: Regional trends, startup hubs
- **Insights**: Location-based success factors

**Use Case**: Understand how geography influences startup success.

---

## 🛠️ Technical Stack

### Data Science & ML
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **statsmodels** - Statistical analysis

### Visualization
- **matplotlib** - Static plots
- **seaborn** - Statistical visualization
- **plotly** - Interactive charts
- **networkx** - Network graphs

### Dashboard
- **Streamlit** - Web application framework
- **streamlit-option-menu** - Navigation components

### Model Interpretation
- **SHAP** - Model explainability

---

## 📈 Key Findings

### Investment Patterns
- **Top Investors**: Aman Gupta (₹5,854L, 144 deals), Namita Thapar (₹4,494L, 121 deals)
- **Most Active Partnership**: Aman-Namita (45 collaborations)
- **Highest Success Industry**: Beauty & Fashion (72.3%)

### Valuation Insights
- **Average Valuation**: ₹1,058 Lakhs
- **Typical Revenue Multiple**: 4.5x - 6.0x
- **Equity Range**: 5% - 15% most common

### Geographic Trends
- **Top States**: Maharashtra (163), Delhi (94), Karnataka (70)
- **Highest Success Rate**: Telangana (74.1%)
- **Emerging Hubs**: Bangalore, Hyderabad, Pune

---

## 📚 Documentation

Detailed documentation for each module is available in the `notebooks/` directory:

- **EDA Report**: Comprehensive exploratory analysis
- **Model Documentation**: ML model architecture and performance
- **API Reference**: Function and class documentation
- **User Guide**: Step-by-step usage instructions

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Data Science Team**
- Project Lead: [Your Name]
- Contributors: [Team Members]

---

## 🙏 Acknowledgments

- Shark Tank India for the inspiration
- The sharks for their investment insights
- All the entrepreneurs who pitched on the show
- Open-source community for amazing tools

---

## 📞 Contact

For questions, suggestions, or collaborations:

- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn]
- **GitHub**: [Your GitHub]

---

## 🔮 Future Enhancements

- [ ] Real-time data integration
- [ ] Sentiment analysis of pitch transcripts
- [ ] Deep learning models for success prediction
- [ ] Mobile app version
- [ ] API for external integrations
- [ ] Advanced time-series forecasting
- [ ] Recommendation system for founders

---

**Built with ❤️ using Python, Streamlit, XGBoost, NetworkX, and Plotly**

*Last Updated: February 2026*
