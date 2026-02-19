# 🚀 Quick Start Guide — Shark Tank India Intelligence Hub

> **Status:** ✅ Production Ready | All 9 phases complete

---

## 1. Environment Setup

```bash
# Clone and enter the project
git clone https://github.com/your-username/sharktank-intelligence-hub.git
cd sharktank-intelligence-hub/shark_tank_intelligence_hub

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install all dependencies
pip install -r requirements.txt
```

---

## 2. Add Your Dataset

Place the raw CSV in the expected location:

```
data/raw/shark_tank_india.csv
```

The dataset should have **702 rows × 80 columns** covering:
- Startup details (name, industry, state/city)
- Financial metrics (revenue, profit, asked amount, equity %)
- Per-shark investment columns (0/1 flags)
- Deal outcomes (offer received, accepted, final terms)

---

## 3. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at **http://localhost:8501**

Navigate the sidebar to access all 6 modules:

| Page | What it does |
|---|---|
| 🎯 Shark Predictor | Input startup details → get per-shark probability scores |
| 💰 Valuation Check | Compare your ask against industry benchmarks |
| 🤝 Shark Networks | Explore co-investment partnerships and influence scores |
| 🏭 Industry Intel | Sector-specific success rates and entry requirements |
| 📋 Deal Decoder | Analyse debt/royalty structures + live cost calculator |
| 🗺️ Geo Insights | State and region investment pattern maps |

---

## 4. Use Production Tools Directly

All tools work standalone — no dashboard required.

### Predict deal outcome
```python
from predict_startup_final import SharkTankPredictorFinal

predictor = SharkTankPredictorFinal()
predictor.explain_prediction(startup_features_df, "My Startup")
# Prints: offer probability, shark recommendations, confidence level
```

### Calculate fair valuation
```python
from valuation_calculator import ValuationCalculator

calc = ValuationCalculator()
calc.calculate_recommended_valuation(
    industry='Technology/Software',
    yearly_revenue=450,    # ₹ Lakhs
    gross_margin=75,
    has_patent=True,
    team_quality=8
)
```

### Get shark recommendations
```python
from shark_recommender import SharkRecommender

recommender = SharkRecommender()
recs = recommender.recommend_sharks({
    'industry': 'Medical/Health',
    'founder_gender': 'female',
    'revenue': 200,
    'stage': 'early'
}, top_n=5)
print(recommender.explain_recommendation(recs[0], startup_profile))
```

### Get deal structure advice
```python
from deal_recommendations import DealStructureRecommender

advisor = DealStructureRecommender()
rec = advisor.recommend_deal_structure({
    'yearly_revenue': 450,
    'gross_margin': 52,
    'cash_burn': False,
    'industry': 'Food and Beverage'
})
print(advisor.explain_recommendation(rec, startup_profile))
```

---

## 5. Run Analysis Notebooks

```bash
jupyter notebook
```

Open notebooks in sequence from the `notebooks/` directory:

| Notebook | Module |
|---|---|
| `01_eda_comprehensive.ipynb` | Exploratory Data Analysis |
| `02_feature_engineering.ipynb` | Feature Engineering |
| `03_shark_predictor.ipynb` | ML Model Training |
| `04_valuation_analysis.ipynb` | Valuation Benchmarking |
| `05_network_analysis.ipynb` | Shark Network Analysis |
| `06_industry_intelligence.ipynb` | Industry Intelligence |
| `07_deal_structure.ipynb` | Deal Structure Decoder |
| `08_geographic_analysis.ipynb` | Geographic Mapping |

---

## 6. Use the Core Library

```python
from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.feature_engineer import FeatureEngineer
from src.analysis.network_analyzer import NetworkAnalyzer
from src.analysis.industry_profiler import IndustryProfiler

# Load and clean data
loader = DataLoader()
df_raw = loader.load_raw_data()

cleaner = DataCleaner()
df_clean = cleaner.clean_dataset(df_raw)
print(cleaner.get_cleaning_report())

# Engineer features
engineer = FeatureEngineer()
df_features = engineer.create_features(df_clean)

# Analyse shark network
analyzer = NetworkAnalyzer()
graph = analyzer.build_shark_network(df_features, shark_columns=[...])
print(analyzer.get_network_statistics())
print(analyzer.get_shark_influence_score())
```

---

## 7. Troubleshooting

**Module import errors**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Missing model files** — ensure these exist before running predictions:
```
models/tuned/best_model_final.pkl
models/tuned/scaler.pkl
models/clean/shark_multilabel_models_clean.pkl
models/clean/shark_multilabel_scaler.pkl
models/deal_structure_predictor.pkl
```

**Dependency conflicts**
```bash
pip install -r requirements.txt --upgrade
```

**Streamlit port already in use**
```bash
streamlit run dashboard/app.py --server.port 8502
```

---

## 8. Enable Logging

Add this to any script to see detailed logs from all modules:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
```

---

*Shark Tank India Intelligence Hub — February 2026*
