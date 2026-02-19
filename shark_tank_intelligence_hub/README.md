# 🦈 Shark Tank India Intelligence Hub

> **Comprehensive Data Science & Business Analytics Platform**  
> Analyzing 702 pitches from Shark Tank India Seasons 1–5 (Dec 2021 – Feb 2026)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-PEP8-black.svg)](https://peps.python.org/pep-0008/)

---

## � Project Overview

This end-to-end data science platform analyzes **702 startup pitches** from Shark Tank India (Seasons 1–5) and delivers **6 production-ready analytical modules** covering machine learning, network analysis, valuation benchmarking, industry intelligence, deal structure analysis, and geographic mapping — all unified in an interactive Streamlit dashboard.

### Dataset at a Glance

| Metric | Value |
|---|---|
| Total Pitches | 702 across 5 seasons |
| Offer Rate | 66.2% received offers |
| Acceptance Rate | 86.2% accepted when offered |
| Total Investment | ₹318.4 Crores |
| Average Deal | ₹79.4 Lakhs for 7.5% equity |
| Industries | 18 categories |
| Geographic Coverage | 29 states |
| Sharks Analyzed | 7 investors |

---

## 🎯 Modules

| # | Module | Description |
|---|---|---|
| 1 | 🎯 **Shark Predictor** | XGBoost multi-label classifier predicting which sharks will invest |
| 2 | 💰 **Valuation Calculator** | Industry-specific benchmarking with revenue/margin/patent multipliers |
| 3 | 🤝 **Shark Network** | NetworkX co-investment graph with centrality & influence scoring |
| 4 | 🏭 **Industry Intelligence** | Sector profiles, success factors, entry requirements for 18 industries |
| 5 | 📋 **Deal Decoder** | Equity / Debt / Royalty structure analysis with effective cost calculator |
| 6 | 🗺️ **Geographic Mapper** | State/region investment patterns and startup hub identification |

---

## 📊 Key Results

### Model Performance
- **Binary Classifier (Deal/No Deal):** 78.5% accuracy, 0.85 AUC-ROC
- **Per-Shark Classifiers:** 75–85% accuracy per shark (7 models)
- **Deal Structure Predictor:** 65.6% accuracy (5 structure types)
- **Valuation Calculator:** Industry-calibrated benchmarks across 18 sectors

### Top Findings
- **Revenue** is the #1 success predictor (18% feature importance)
- **Aman Gupta** is the most active shark — 143 deals, ₹5,854L invested
- **Strongest partnership:** Namita Thapar ↔ Aman Gupta (45 co-investments)
- **72.3%** of deals are pure equity; only 9% include royalty
- **Maharashtra** dominates with 163 pitches (23.2% of all pitches)
- **Female founders** have a statistical advantage in 6 out of 10 top industries
- **Technology** commands the highest revenue multiples (27.9x median)

### Valuation Multiples by Industry
| Industry | Median Revenue Multiple |
|---|---|
| Medical / Health | 36.0x |
| Technology / Software | 29.2x |
| Food & Beverage | 12.1x |
| Beauty / Fashion | 8.4x |

---

## 📁 Project Structure

```
shark_tank_intelligence_hub/
│
├── data/
│   ├── raw/                              # Original CSV (702 rows × 80 cols)
│   └── processed/                        # Cleaned & feature-engineered data
│       ├── processed_data_full.csv
│       ├── processed_data_with_valuation_metrics.csv
│       └── processed_data_with_deal_structures.csv
│
├── notebooks/                            # Analysis notebooks (run in order)
│   ├── 01_eda_comprehensive.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_shark_predictor.ipynb
│   ├── 04_valuation_analysis.ipynb
│   ├── 05_network_analysis.ipynb
│   ├── 06_industry_intelligence.ipynb
│   ├── 07_deal_structure.ipynb
│   └── 08_geographic_analysis.ipynb
│
├── src/                                  # Core library modules
│   ├── data/
│   │   ├── loader.py                     # Data loading utilities
│   │   ├── cleaner.py                    # Cleaning pipeline
│   │   └── feature_engineer.py           # Feature creation (74 features)
│   ├── models/
│   │   ├── shark_predictor.py            # XGBoost multi-label classifier
│   │   ├── valuation_model.py            # Random Forest equity regressor
│   │   └── model_explainer.py            # Feature importance & SHAP
│   └── analysis/
│       ├── eda_utils.py                  # EDA & visualization utilities
│       ├── network_analyzer.py           # NetworkX graph analysis
│       ├── industry_profiler.py          # Industry benchmarks
│       ├── deal_decoder.py               # Deal structure patterns
│       └── geo_mapper.py                 # Geographic analysis
│
├── models/                               # Serialized trained models
│   ├── tuned/
│   │   ├── best_model_final.pkl          # Binary classifier (78.5% acc)
│   │   └── scaler.pkl
│   ├── clean/
│   │   ├── shark_multilabel_models_clean.pkl  # 7 shark classifiers
│   │   └── shark_multilabel_scaler.pkl
│   └── deal_structure_predictor.pkl      # Deal type classifier
│
├── dashboard/                            # Streamlit multi-page app
│   ├── app.py                            # Landing page
│   ├── pages/
│   │   ├── 1_🎯_Shark_Predictor.py
│   │   ├── 2_💰_Valuation_Check.py
│   │   ├── 3_🤝_Shark_Networks.py
│   │   ├── 4_🏭_Industry_Intel.py
│   │   ├── 5_📋_Deal_Decoder.py
│   │   └── 6_🗺️_Geo_Insights.py
│   └── utils/
│       └── helpers.py
│
├── reports/                              # Generated reports & visualizations
│   ├── figures/                          # 25+ charts and graphs
│   ├── VALUATION_INSIGHTS_REPORT.md
│   ├── INDUSTRY_INTELLIGENCE_REPORT.md
│   ├── PHASE4_ML_MODEL_SUMMARY.md
│   ├── shark_profiles.json
│   ├── industry_benchmarks.csv
│   └── state_statistics.csv
│
├── predict_startup_final.py              # Production prediction API
├── valuation_calculator.py              # Production valuation tool
├── shark_recommender.py                 # Production shark matcher
├── deal_recommendations.py             # Production deal advisor
├── requirements.txt                     # Python dependencies
├── config.yaml                          # Project configuration
└── QUICKSTART.md                        # Quick start guide
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/sharktank-intelligence-hub.git
cd sharktank-intelligence-hub/shark_tank_intelligence_hub

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your dataset
# Copy shark_tank_india.csv → data/raw/shark_tank_india.csv
```

### Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`

### Use Production Tools Directly

```python
# Predict deal outcome for a new startup
from predict_startup_final import SharkTankPredictorFinal

predictor = SharkTankPredictorFinal()
result = predictor.predict(startup_features_df)
# → offer_probability, recommended_sharks, confidence

# Calculate fair valuation
from valuation_calculator import ValuationCalculator

calc = ValuationCalculator()
calc.calculate_recommended_valuation(
    industry='Technology/Software',
    yearly_revenue=450,   # ₹ Lakhs
    gross_margin=75,
    has_patent=True,
    team_quality=8
)
# → ₹X recommended valuation with ±20% range

# Get shark recommendations
from shark_recommender import SharkRecommender

recommender = SharkRecommender()
recommender.recommend_sharks({
    'industry': 'Medical/Health',
    'founder_gender': 'female',
    'revenue': 200,
    'stage': 'early'
}, top_n=5)
# → Ranked shark combinations with synergy scores

# Get deal structure advice
from deal_recommendations import DealStructureRecommender

advisor = DealStructureRecommender()
advisor.recommend_deal_structure({
    'yearly_revenue': 450,
    'gross_margin': 52,
    'cash_burn': False,
    'industry': 'Food and Beverage'
})
# → Pure Equity / Debt+Equity / Royalty+Equity with reasoning
```

### Run Analysis Notebooks

```bash
jupyter notebook
# Open notebooks/ and run in sequence: 01 → 02 → 03 → ... → 08
```

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| **Data Processing** | pandas, numpy, scipy |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM |
| **Visualization** | matplotlib, seaborn, plotly |
| **Network Analysis** | NetworkX |
| **Dashboard** | Streamlit |
| **Explainability** | SHAP |
| **Statistical Analysis** | statsmodels |
| **Serialization** | pickle, joblib |

---

## 📈 Production Tools

| Tool | File | Description |
|---|---|---|
| **Startup Predictor** | `predict_startup_final.py` | Binary deal prediction + per-shark probability scores |
| **Valuation Calculator** | `valuation_calculator.py` | Revenue-based & pre-revenue valuation with multipliers |
| **Shark Recommender** | `shark_recommender.py` | Optimal shark combinations with synergy scoring |
| **Deal Advisor** | `deal_recommendations.py` | Rule-based deal structure recommendation engine |

---

## � Reports & Documentation

- [`QUICKSTART.md`](QUICKSTART.md) — Setup and usage guide
- [`reports/VALUATION_INSIGHTS_REPORT.md`](reports/VALUATION_INSIGHTS_REPORT.md) — Valuation analysis
- [`reports/INDUSTRY_INTELLIGENCE_REPORT.md`](reports/INDUSTRY_INTELLIGENCE_REPORT.md) — Industry deep dive
- [`reports/PHASE4_ML_MODEL_SUMMARY.md`](reports/PHASE4_ML_MODEL_SUMMARY.md) — ML model details
- [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) — Complete project summary (all 9 phases)
- [`notebooks/`](notebooks/) — Jupyter analysis notebooks

---

## 👤 Author

**Apoorv Thite**
- LinkedIn: [linkedin.com/in/apoorvthite](https://linkedin.com/in/apoorvthite)
- GitHub: [github.com/apoorvthite](https://github.com/apoorvthite)
- Email: apoorv@example.com

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## � Future Enhancements

- [ ] Real-time data integration via web scraping
- [ ] NLP sentiment analysis of pitch transcripts
- [ ] Deep learning (LSTM) for time-series investment forecasting
- [ ] REST API deployment (FastAPI)
- [ ] Mobile-responsive dashboard
- [ ] Automated weekly insights report generation

---

*Built with Python · Streamlit · XGBoost · NetworkX · Plotly*  
*Last Updated: February 2026*
