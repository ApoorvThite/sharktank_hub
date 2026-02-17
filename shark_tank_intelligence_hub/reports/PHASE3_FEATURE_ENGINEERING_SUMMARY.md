# 🛠️ PHASE 3: DATA PREPROCESSING & FEATURE ENGINEERING - COMPLETE

**Date:** February 16, 2026  
**Status:** ✅ COMPLETED  
**Dataset:** 702 pitches, Seasons 1-5

---

## 📊 EXECUTIVE SUMMARY

Successfully created **47 engineered features** from the raw Shark Tank India dataset, transforming 80 original columns into **137 total columns** (80 original + 47 new + 10 targets). The processed data is now ready for machine learning model training.

### Key Achievements
- ✅ **47 new features** engineered across 7 categories
- ✅ **82 features** selected for modeling (removed redundant/target variables)
- ✅ **3 target types** created: binary, multi-label (7 sharks), regression
- ✅ **561/141 train/test split** (80/20) with stratification by industry
- ✅ **All data saved** to `data/processed/` directory

---

## 🧹 SECTION 1: DATA CLEANING

### Missing Value Treatment

**Shark Investment Columns (24 columns)**
- Filled with **0** (indicates no investment)
- Columns: `{Shark} Investment Amount`, `{Shark} Investment Equity`, `{Shark} Debt Amount`
- Sharks: Namita, Vineeta, Anupam, Aman, Peyush, Ritesh, Amit, Guest

**Financial Metrics (5 columns)**
- Imputed with **industry median**
- Columns: Yearly Revenue, Monthly Sales, Gross Margin, Net Margin, EBITDA
- Strategy: Preserves industry-specific patterns

**Deal Columns (7 columns)**
- Filled with **0** (no deal = no values)
- Columns: Total Deal Amount, Total Deal Equity, Total Deal Debt, Debt Interest, Royalty Percentage, Royalty Recouped Amount, Advisory Shares Equity

**SKUs**
- Filled with **median** value

### Data Type Conversions

**Binary Encoding**
- `Cash Burn`: Yes/No → 1/0
- `Has Patents`: Yes/No → 1/0
- `Bootstrapped`: Yes/No → 1/0

**Numeric Conversion**
- Ensured all financial columns are numeric
- Handled errors with coercion to 0

### Outlier Detection (Flagged, Not Removed)

**`is_outlier_valuation`**
- Flags valuations > Q3 + 1.5×IQR
- Preserves extreme but valid cases

**`is_high_revenue`**
- Flags startups with revenue > ₹1000L
- Identifies established businesses

---

## 💡 SECTION 2: FEATURE ENGINEERING (47 FEATURES)

### 1️⃣ Financial Health Indicators (10 features)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `revenue_per_sku` | Yearly Revenue / (SKUs + 1) | Product efficiency |
| `monthly_to_yearly_ratio` | (Monthly Sales × 12) / (Yearly Revenue + 1) | Revenue consistency |
| `profit_margin_gap` | Gross Margin - Net Margin | Operating efficiency |
| `profitability_score` | (Net Margin × Yearly Revenue) / 1000 | Absolute profitability |
| `ebitda_margin` | EBITDA / (Yearly Revenue + 1) | Operating margin |
| `burn_rate` | Cash Burn × Monthly Sales | Cash consumption |
| `runway_months` | Yearly Revenue / (burn_rate + 1) | Survival duration |
| `is_pre_revenue` | Yearly Revenue == 0 | Binary flag |
| `revenue_category` | Ordinal 0-4 | Revenue tier |
| `financial_health_score` | Composite 0-5 | Overall health |

### 2️⃣ Deal Structure Indicators (8 features)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `revenue_multiple` | Valuation / (Revenue + 1) | Valuation metric |
| `ask_percentage` | (Ask Amount / Valuation) × 100 | Equity ask |
| `valuation_reasonableness` | Revenue multiple / Industry median | Relative valuation |
| `expected_equity_dilution` | Original Offered Equity | Dilution expectation |
| `deal_size_category` | Ordinal 0-4 | Deal tier |
| `valuation_to_ask_ratio` | Valuation / Ask Amount | Capital efficiency |
| `is_reasonable_valuation` | Reasonableness ≤ 2 | Binary flag |
| `deal_complexity_score` | Count of debt/royalty/advisory | Deal complexity |

### 3️⃣ Team Composition Features (7 features)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `team_size` | Number of Presenters | Team scale |
| `male_ratio` | Male / Team Size | Gender composition |
| `female_ratio` | Female / Team Size | Gender composition |
| `gender_diversity_score` | 1 - abs(male_ratio - female_ratio) | Diversity metric |
| `is_solo_founder` | Team Size == 1 | Binary flag |
| `is_couple` | Couple Presenters | Binary flag |
| `has_female_founder` | Female > 0 | Binary flag |

### 4️⃣ Innovation Indicators (4 features)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `has_patent` | Has Patents (binary) | IP protection |
| `is_bootstrapped` | Bootstrapped (binary) | Self-funded |
| `sku_count` | SKUs | Product range |
| `innovation_score` | (patent × 2) + bootstrapped | Composite 0-3 |

### 5️⃣ Shark Affinity Scores (7 features)

**One feature per shark:** `{shark}_industry_fit`

**Calculation:**
```python
affinity = (shark investments in industry) / (total shark investments)
```

**Sharks:** Namita, Aman, Anupam, Peyush, Vineeta, Ritesh, Amit

**Purpose:** Predicts shark interest based on historical industry preferences

### 6️⃣ Industry Context Features (5 features)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `industry_avg_success_rate` | Mean(Received Offer) by Industry | Industry benchmark |
| `industry_avg_equity` | Mean(Total Deal Equity) by Industry | Typical dilution |
| `industry_median_valuation` | Median(Valuation) by Industry | Valuation benchmark |
| `industry_pitch_count` | Count by Industry | Industry popularity |
| `industry_competition_index` | Pitch Count / Total | Competition level |

### 7️⃣ Geographic Features (4 features)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `state_success_rate` | Mean(Received Offer) by State | State benchmark |
| `is_metro` | State in [MH, DL, KA] | Metro flag |
| `state_pitch_density` | Count by State | State activity |
| `geographic_diversity_score` | Pitch Density / Total | Normalized density |

---

## 🎯 SECTION 3: TARGET VARIABLE ENGINEERING

### Binary Classification Targets (2)

**`got_offer`**
- 465/702 (66.2%) received offers
- Primary classification target

**`accepted_offer`**
- 401/702 (57.1%) accepted offers
- Secondary classification target

### Multi-Label Classification Targets (7)

Individual shark investment predictions:

| Shark | Deals | Investment Rate |
|-------|-------|-----------------|
| `aman_invested` | 143 | 20.4% |
| `namita_invested` | 121 | 17.2% |
| `anupam_invested` | 108 | 15.4% |
| `peyush_invested` | 103 | 14.7% |
| `vineeta_invested` | 91 | 13.0% |
| `ritesh_invested` | 53 | 7.5% |
| `amit_invested` | 36 | 5.1% |

### Regression Target (1)

**`equity_dilution`**
- Mean: 0.79%
- Std: 7.27%
- Predicts difference between final and initial equity

---

## 🔍 SECTION 4: FEATURE SELECTION

### Selection Process

**Total numeric columns:** 121  
**Excluded columns:** 39  
- Target variables (10)
- Identifiers (3): Season, Episode, Pitch Number
- Individual shark investments (26): Replaced by affinity scores

**Selected features:** 82

### Top 20 Features by Correlation with `got_offer`

| Rank | Feature | Correlation |
|------|---------|-------------|
| 1 | Total Deal Amount | 0.5338 |
| 2 | Total Deal Equity | 0.4118 |
| 3 | deal_complexity_score | 0.3306 |
| 4 | state_success_rate | 0.2831 |
| 5 | Debt Interest | 0.2151 |
| 6 | Total Deal Debt | 0.1693 |
| 7 | industry_avg_success_rate | 0.1603 |
| 8 | valuation_to_ask_ratio | 0.1500 |
| 9 | Royalty Percentage | 0.1464 |
| 10 | Royalty Recouped Amount | 0.1437 |
| 11 | revenue_category | 0.1286 |
| 12 | peyush_industry_fit | 0.0953 |
| 13 | namita_industry_fit | 0.0902 |
| 14 | Advisory Shares Equity | 0.0876 |
| 15 | profitability_score | 0.0836 |
| 16 | aman_industry_fit | 0.0832 |
| 17 | ritesh_industry_fit | 0.0773 |
| 18 | anupam_industry_fit | 0.0770 |
| 19 | Gross Margin | 0.0751 |
| 20 | runway_months | 0.0742 |

**Key Insights:**
- Deal-related features show strongest correlation
- State and industry context features are important
- Shark affinity scores show moderate correlation
- Financial health indicators contribute moderately

---

## ✂️ SECTION 5: TRAIN/TEST SPLIT

### Split Configuration

**Method:** Stratified by Industry  
**Test Size:** 20%  
**Random State:** 42 (reproducible)

### Split Results

| Dataset | Samples | Percentage | Offers | Offer Rate |
|---------|---------|------------|--------|------------|
| **Training** | 561 | 79.9% | 362 | 64.5% |
| **Test** | 141 | 20.1% | 103 | 73.0% |
| **Total** | 702 | 100% | 465 | 66.2% |

**Note:** Test set has slightly higher offer rate (73.0% vs 64.5%) due to stratification maintaining industry distribution rather than target distribution.

---

## 💾 SECTION 6: SAVED FILES

### Train/Test Splits (8 files)

| File | Shape | Description |
|------|-------|-------------|
| `X_train.csv` | (561, 82) | Training features |
| `X_test.csv` | (141, 82) | Test features |
| `y_train_binary.csv` | (561, 1) | Training binary target |
| `y_test_binary.csv` | (141, 1) | Test binary target |
| `y_train_multilabel.csv` | (561, 7) | Training multi-label targets |
| `y_test_multilabel.csv` | (141, 7) | Test multi-label targets |
| `y_train_regression.csv` | (561, 1) | Training regression target |
| `y_test_regression.csv` | (141, 1) | Test regression target |

### Additional Files (3 files)

| File | Shape | Description |
|------|-------|-------------|
| `processed_data_full.csv` | (702, 137) | Complete processed dataset |
| `feature_list.csv` | (82, 2) | Feature names and data types |
| `feature_importance_preliminary.csv` | (82, 2) | Correlation-based importance |

**Total Files:** 11  
**Location:** `data/processed/`

---

## 📈 FEATURE ENGINEERING IMPACT

### Before vs After

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Columns** | 80 | 137 | +57 (+71%) |
| **Modeling Features** | ~60 | 82 | +22 (+37%) |
| **Target Variables** | 2 | 10 | +8 (+400%) |
| **Missing Values** | 31,445 | 0 | -31,445 (-100%) |

### Feature Categories Distribution

```
Financial Health:     10 features (21%)
Deal Structure:        8 features (17%)
Team Composition:      7 features (15%)
Shark Affinity:        7 features (15%)
Industry Context:      5 features (11%)
Innovation:            4 features (9%)
Geographic:            4 features (9%)
Outlier Flags:         2 features (4%)
────────────────────────────────────
TOTAL:                47 features (100%)
```

---

## 🎯 KEY INSIGHTS FROM FEATURE ENGINEERING

### 1. Deal Complexity Matters
- `deal_complexity_score` has 3rd highest correlation (0.33)
- Deals with debt/royalty/advisory terms more likely to succeed
- Suggests sharks prefer structured deals over simple equity

### 2. Geography is Significant
- `state_success_rate` has 4th highest correlation (0.28)
- Location impacts success independent of other factors
- Metro states (MH, DL, KA) show different patterns

### 3. Industry Context is Crucial
- `industry_avg_success_rate` correlates at 0.16
- Industry benchmarks help normalize expectations
- Competition index shows saturation effects

### 4. Shark Preferences are Predictable
- All 7 shark affinity scores show positive correlation
- Peyush (0.095) and Namita (0.090) strongest
- Industry fit predicts individual shark interest

### 5. Financial Health is Moderate Predictor
- `profitability_score` correlates at 0.084
- `revenue_category` correlates at 0.129
- Revenue matters, but not as much as deal structure

---

## 🚀 NEXT STEPS: PHASE 4 - ML MODEL TRAINING

### Recommended Models

**1. Binary Classification (got_offer)**
- XGBoost Classifier
- Random Forest Classifier
- Logistic Regression (baseline)

**2. Multi-Label Classification (7 sharks)**
- MultiOutputClassifier with XGBoost
- Binary Relevance with Random Forest
- Classifier Chains

**3. Regression (equity_dilution)**
- XGBoost Regressor
- Random Forest Regressor
- Linear Regression (baseline)

### Model Training Checklist

- [ ] Train baseline models (Logistic, Linear)
- [ ] Train ensemble models (RF, XGBoost)
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Cross-validation (5-fold)
- [ ] Feature importance analysis
- [ ] Model evaluation (accuracy, precision, recall, F1, RMSE)
- [ ] Save trained models
- [ ] Generate prediction examples

---

## 📝 TECHNICAL NOTES

### Dependencies Used
```python
pandas==2.2.3
numpy==2.2.6
scikit-learn==1.7.2
scipy==1.15.3
```

### Reproducibility
- Random state: 42 (all splits)
- Stratification: By Industry
- Missing value strategy: Documented and consistent

### Data Quality
- No missing values in final dataset
- All features numeric (ready for ML)
- Outliers flagged but retained
- Industry distribution preserved in train/test

---

## ✅ PHASE 3 COMPLETION CHECKLIST

- [x] Data cleaning completed
- [x] 47 features engineered
- [x] 10 target variables created
- [x] 82 features selected for modeling
- [x] Train/test split (80/20) completed
- [x] All data saved to `data/processed/`
- [x] Feature importance analysis completed
- [x] Documentation created
- [x] Ready for Phase 4: ML Model Training

---

**Phase 3 Status:** ✅ **COMPLETE**  
**Next Phase:** Phase 4 - ML Model Training  
**Estimated Time:** 2-3 days

---

*Generated: February 16, 2026*  
*Shark Tank India Intelligence Hub - Phase 3*
