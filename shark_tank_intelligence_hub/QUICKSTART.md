# 🚀 Quick Start Guide

## Shark Tank India Intelligence Hub

### Step 1: Setup Environment

```bash
# Navigate to project directory
cd shark_tank_intelligence_hub

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Data

1. Place your `shark_tank_india.csv` file in `data/raw/` directory
2. The dataset should contain 702 rows with columns for:
   - Startup details (name, industry, location)
   - Financial metrics (revenue, profit, asked amount, equity)
   - Shark decisions (individual shark columns)
   - Deal outcomes (accepted, amount invested, equity taken)

### Step 3: Run Analysis Notebooks

Open Jupyter and run notebooks in order:

```bash
jupyter notebook
```

**Recommended sequence:**
1. `01_eda_comprehensive.ipynb` - Understand the data
2. `02_feature_engineering.ipynb` - Create features
3. `03_shark_predictor.ipynb` - Build ML model
4. `04_valuation_analysis.ipynb` - Analyze valuations
5. `05_network_analysis.ipynb` - Shark networks
6. `06_industry_intelligence.ipynb` - Industry insights
7. `07_deal_structure.ipynb` - Deal analysis
8. `08_geographic_analysis.ipynb` - Geographic patterns

### Step 4: Launch Dashboard

```bash
# From project root
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`

### Step 5: Explore Modules

Navigate through the dashboard pages:

- **🎯 Shark Predictor** - Predict which sharks will invest
- **💰 Valuation Check** - Benchmark your valuation
- **🤝 Shark Networks** - Explore partnerships
- **🏭 Industry Intel** - Industry-specific insights
- **📋 Deal Decoder** - Analyze deal structures
- **🗺️ Geo Insights** - Geographic patterns

---

## 📊 Phase 1: Next Steps

Once the structure is ready, Phase 1 will involve:

1. **Data Collection & Cleaning**
   - Obtain the actual Shark Tank India dataset
   - Clean and validate data
   - Handle missing values

2. **Exploratory Data Analysis**
   - Complete `01_eda_comprehensive.ipynb`
   - Generate visualizations
   - Document key findings

3. **Feature Engineering**
   - Complete `02_feature_engineering.ipynb`
   - Create all 35+ features
   - Save processed data

4. **Model Development**
   - Train shark predictor model
   - Optimize hyperparameters
   - Validate performance

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Module import errors
```bash
# Solution: Add project to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/shark_tank_intelligence_hub"
```

**Issue**: Missing dependencies
```bash
# Solution: Reinstall requirements
pip install -r requirements.txt --upgrade
```

**Issue**: Streamlit not found
```bash
# Solution: Install streamlit explicitly
pip install streamlit==1.27.0
```

---

## 📞 Support

For issues or questions:
- Check the main `README.md`
- Review notebook documentation
- Check `config.yaml` for configuration options

---

**Ready to analyze Shark Tank India data! 🦈📊**
