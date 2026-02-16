# 📊 SHARK TANK INDIA - VISUALIZATION & DATA INDEX

## Generated Files Summary

**Total Files Created:** 29
- **Visualizations:** 16 PNG charts
- **Data Tables:** 11 CSV files  
- **Reports:** 2 comprehensive documents

---

## 🎨 VISUALIZATIONS (reports/figures/)

### Financial Analysis

**1. Financial Distributions** (`01_financial_distributions.png`)
- 8-panel grid showing distributions of key financial metrics
- Includes: Revenue, Sales, Margins, Ask Amount, Valuation, Deal Amount, Equity
- Shows mean and median lines for each metric

**2. Outlier Box Plots** (`02_outlier_boxplots.png`)
- Box plots for outlier detection across 8 financial variables
- Displays outlier counts for each metric
- Helps identify extreme values in the dataset

**14. Deal Amount Distribution** (`14_deal_amount_distribution.png`)
- Histogram and box plot of deal amounts
- Mean: ₹79.4L, Median: ₹60L
- Shows right-skewed distribution

**16. Equity Distribution** (`16_equity_distribution.png`)
- Distribution of equity dilution in deals
- Mean: 7.5%, Median varies by season
- Shows typical equity ranges

---

### Industry Analysis

**3. Industry Distribution** (`03_industry_distribution.png`)
- Bar chart + pie chart showing top 15 industries
- Food & Beverage leads with 154 pitches (21.9%)
- Beauty/Fashion second with 141 pitches (20.1%)

**7. Industry Success Heatmap** (`07_industry_success_heatmap.png`)
- Multi-metric heatmap for top 15 industries
- Shows: Success Rate, Avg Deal, Median Revenue, Avg Valuation
- Agriculture shows highest success rate (85.7%)

**15. Industry vs State Heatmap** (`15_industry_state_heatmap.png`)
- Cross-tabulation of top 10 industries vs top 10 states
- Shows geographic concentration by industry
- Identifies industry-location patterns

---

### Geographic Analysis

**4. Geographic Distribution** (`04_geographic_distribution.png`)
- Horizontal bar chart of top 15 states
- Maharashtra leads with 163 pitches (23.2%)
- Shows percentage distribution

**13. State Success Rates** (`13_state_success_rates.png`)
- Success rates and pitch volumes by state
- Gujarat highest success rate (73.7%)
- Compares top 10 states

---

### Success Metrics

**5. Success Metrics** (`05_success_metrics.png`)
- Two pie charts showing:
  - Received Offer Rate: 66.2%
  - Deal Acceptance Rate: 86.2%

**6. Revenue vs Success** (`06_revenue_vs_success.png`)
- Success rates by revenue category
- Pre-revenue: 55.6% → High revenue: 78.4%
- Shows clear correlation between revenue and success

---

### Shark Analysis

**10. Shark Performance** (`10_shark_performance.png`)
- Total investment and deal count by shark
- Aman leads with ₹5,854L invested, 143 deals
- Namita second with ₹4,493L, 121 deals

---

### Temporal Analysis

**9. Season Trends** (`09_season_trends.png`)
- 4-panel chart showing evolution across seasons:
  - Success rate trends
  - Average deal size growth (₹57.5L → ₹135.6L)
  - Pitch volume by season
  - Valuation trends

---

### Statistical Analysis

**8. Correlation Matrix** (`08_correlation_matrix.png`)
- Heatmap of correlations between key financial metrics
- Identifies strong relationships (>0.7)
- Revenue ↔ Monthly Sales: 0.89

**12. Skewness Analysis** (`12_skewness_analysis.png`)
- Top 15 right-skewed and left-skewed variables
- Helps identify distribution patterns
- Guides transformation strategies

**11. Missing Data Analysis** (`11_missing_data_analysis.png`)
- Top 25 variables with missing data
- Color-coded by severity (red >50%, orange >20%)
- Total: 31,445 missing values across 70 columns

---

## 📊 DATA TABLES (reports/tables/)

### Industry Tables

**1. industry_distribution.csv**
- Industry name, pitch count, percentage
- 15 rows covering top industries

**2. industry_metrics.csv**
- Success rate, avg deal, median revenue, avg valuation by industry
- Top 15 industries by success rate

**3. industry_state_distribution.csv**
- Cross-tabulation of industries vs states
- Shows geographic concentration patterns

---

### Geographic Tables

**4. state_distribution.csv**
- State name, pitch count, percentage
- 15 rows covering top states

**5. state_success_rates.csv**
- State, offers received, total pitches, success rate
- Top 10 states by volume

---

### Performance Tables

**6. shark_performance.csv**
- Shark name, total investment, number of deals, avg investment
- 7 sharks analyzed

**7. season_analysis.csv**
- Season-wise: offers, pitches, avg deal, avg equity, avg valuation, success rate
- 5 seasons covered

**8. revenue_vs_success.csv**
- Revenue category, offers, total pitches, success rate
- 4 categories: Pre-revenue, Low, Medium, High

---

### Statistical Tables

**9. correlation_matrix.csv**
- Full correlation matrix of key financial metrics
- 9x9 matrix with correlation coefficients

**10. skewness_analysis.csv**
- Variable name and skewness value for all 61 numerical variables
- Sorted by skewness (highest to lowest)

**11. missing_data_summary.csv**
- Column name, missing count, missing percentage
- 70 columns with missing data

---

## 📄 REPORTS (reports/)

**1. eda_insights_report.txt**
- Comprehensive insights summary
- 7 key findings documented
- Actionable recommendations for founders, investors, researchers

**2. PHASE2_EDA_GUIDE.md**
- Complete implementation guide
- Code examples and usage instructions
- Troubleshooting tips

---

## 🎯 HOW TO USE THESE FILES

### Viewing Visualizations
```bash
# Navigate to figures directory
cd reports/figures

# Open specific visualization
open 01_financial_distributions.png

# Or open all at once (macOS)
open *.png
```

### Analyzing Data Tables
```python
import pandas as pd

# Load any table
industry_data = pd.read_csv('reports/tables/industry_distribution.csv')
shark_stats = pd.read_csv('reports/tables/shark_performance.csv')

# Analyze
print(industry_data.head())
print(shark_stats.describe())
```

### Using in Presentations
- All visualizations are high-resolution (300 DPI)
- Ready for PowerPoint, reports, or publications
- Tables can be imported into Excel or Google Sheets

---

## 📈 KEY INSIGHTS FROM VISUALIZATIONS

### From Financial Distributions (01)
- Revenue is highly right-skewed (median ₹240L, mean ₹672L)
- Deal amounts cluster around ₹50-100L range
- Equity typically 5-15%

### From Industry Analysis (03, 07)
- Food & Beverage dominates volume but not success rate
- Agriculture shows highest success (85.7%) but low volume
- Technology commands higher valuations

### From Geographic Analysis (04, 13)
- Maharashtra, Delhi, Karnataka = 46.6% of all pitches
- Gujarat shows best success rate among top states (73.7%)
- Tier-2 cities emerging

### From Revenue vs Success (06)
- **Critical Finding:** Revenue is strongest predictor
- Pre-revenue: 55.6% success
- High revenue (>₹1000L): 78.4% success
- 23% improvement with high revenue

### From Season Trends (09)
- Deal sizes growing 2.4x (₹57.5L → ₹135.6L)
- Equity dilution decreasing (better for founders)
- Market maturing rapidly

### From Shark Performance (10)
- Aman most active (143 deals, ₹5,854L)
- Namita focused on health/beauty
- Ritesh highest participation rate (36.3%)

---

## 🚀 NEXT STEPS

These visualizations and tables are ready for:

1. **Presentations** - Use in pitch decks or investor reports
2. **Analysis** - Further statistical analysis in Python/R
3. **Dashboards** - Import into Tableau, Power BI, or Streamlit
4. **Publications** - Academic papers or blog posts
5. **Decision Making** - Data-driven startup strategy

---

**Generated:** February 16, 2026  
**Dataset:** 702 pitches, Seasons 1-5  
**Analysis Tool:** Python (pandas, matplotlib, seaborn)
