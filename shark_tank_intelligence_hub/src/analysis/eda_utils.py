import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional

class EDAAnalyzer:
    """Comprehensive EDA utilities for Shark Tank India analysis"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def get_summary_stats(self) -> Dict:
        """Get comprehensive dataset summary statistics"""
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numerical_vars': len(self.numerical_cols),
            'categorical_vars': len(self.categorical_cols),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum()
        }
    
    def analyze_distribution(self, column: str) -> Dict:
        """Analyze distribution of a numerical variable"""
        data = self.df[column].dropna()
        
        return {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75),
            'skewness': skew(data),
            'kurtosis': kurtosis(data),
            'missing_pct': (self.df[column].isnull().sum() / len(self.df) * 100)
        }
    
    def detect_outliers(self, column: str, method: str = 'iqr') -> Tuple[int, pd.Series]:
        """Detect outliers using IQR or Z-score method"""
        data = self.df[column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
        else:  # z-score
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = data[z_scores > 3]
        
        return len(outliers), outliers
    
    def analyze_categorical(self, column: str) -> pd.DataFrame:
        """Analyze categorical variable distribution"""
        freq = self.df[column].value_counts()
        pct = self.df[column].value_counts(normalize=True) * 100
        
        return pd.DataFrame({
            'Count': freq,
            'Percentage': pct.round(2)
        })
    
    def correlation_analysis(self, columns: Optional[List[str]] = None, threshold: float = 0.5) -> pd.DataFrame:
        """Find strong correlations between variables"""
        if columns is None:
            columns = self.numerical_cols
        
        corr_matrix = self.df[columns].corr()
        
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    strong_corr.append({
                        'Variable_1': corr_matrix.columns[i],
                        'Variable_2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        return pd.DataFrame(strong_corr).sort_values('Correlation', ascending=False, key=abs)
    
    def missing_data_analysis(self) -> pd.DataFrame:
        """Comprehensive missing data analysis"""
        missing_summary = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2),
            'Data_Type': self.df.dtypes
        })
        
        missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
        missing_summary = missing_summary.sort_values('Missing_Percentage', ascending=False)
        
        return missing_summary
    
    def bivariate_analysis(self, var1: str, var2: str, success_col: str = 'received_offer') -> pd.DataFrame:
        """Analyze relationship between two variables and success"""
        if var1 in self.categorical_cols and var2 in self.categorical_cols:
            # Both categorical
            return pd.crosstab(self.df[var1], self.df[var2], normalize='index') * 100
        
        elif var1 in self.categorical_cols:
            # var1 categorical, var2 numerical
            return self.df.groupby(var1)[var2].agg(['mean', 'median', 'std', 'count'])
        
        else:
            # Both numerical - correlation
            return self.df[[var1, var2]].corr()
    
    def revenue_success_analysis(self, revenue_col: str = 'yearly_revenue', 
                                 success_col: str = 'received_offer') -> pd.DataFrame:
        """Analyze revenue vs success rate"""
        if revenue_col not in self.df.columns or success_col not in self.df.columns:
            return pd.DataFrame()
        
        # Create revenue categories
        df_temp = self.df.copy()
        df_temp['revenue_category'] = pd.cut(df_temp[revenue_col], 
                                             bins=[-np.inf, 0, 100, 1000, np.inf],
                                             labels=['Pre-revenue', 'Low (1-100L)', 
                                                    'Medium (100-1000L)', 'High (>1000L)'])
        
        analysis = df_temp.groupby('revenue_category')[success_col].agg(['sum', 'count'])
        analysis['success_rate'] = (analysis['sum'] / analysis['count'] * 100).round(1)
        
        return analysis
    
    def industry_performance(self, industry_col: str = 'industry',
                           success_col: str = 'received_offer',
                           top_n: int = 15) -> pd.DataFrame:
        """Analyze industry-wise performance metrics"""
        if industry_col not in self.df.columns:
            return pd.DataFrame()
        
        industry_stats = self.df.groupby(industry_col).agg({
            success_col: lambda x: (x.sum() / len(x) * 100) if success_col in self.df.columns else 0,
            'total_deal_amount': 'mean' if 'total_deal_amount' in self.df.columns else lambda x: 0,
            'yearly_revenue': 'median' if 'yearly_revenue' in self.df.columns else lambda x: 0,
            'valuation_requested': 'mean' if 'valuation_requested' in self.df.columns else lambda x: 0
        }).round(2)
        
        industry_stats.columns = ['Success_Rate', 'Avg_Deal', 'Median_Revenue', 'Avg_Valuation']
        industry_stats['Pitch_Count'] = self.df[industry_col].value_counts()
        
        return industry_stats.nlargest(top_n, 'Success_Rate')
    
    def temporal_analysis(self, time_col: str = 'season') -> pd.DataFrame:
        """Analyze trends over time (seasons)"""
        if time_col not in self.df.columns:
            return pd.DataFrame()
        
        temporal_stats = self.df.groupby(time_col).agg({
            'received_offer': ['sum', 'count'] if 'received_offer' in self.df.columns else lambda x: 0,
            'total_deal_amount': 'mean' if 'total_deal_amount' in self.df.columns else lambda x: 0,
            'total_deal_equity': 'mean' if 'total_deal_equity' in self.df.columns else lambda x: 0,
            'valuation_requested': 'mean' if 'valuation_requested' in self.df.columns else lambda x: 0
        }).round(2)
        
        if 'received_offer' in self.df.columns:
            temporal_stats.columns = ['Offers', 'Pitches', 'Avg_Deal', 'Avg_Equity', 'Avg_Valuation']
            temporal_stats['Success_Rate'] = (temporal_stats['Offers'] / temporal_stats['Pitches'] * 100).round(1)
        
        return temporal_stats
    
    def generate_insights_report(self) -> str:
        """Generate comprehensive insights report"""
        summary = self.get_summary_stats()
        
        report = f"""
{'='*80}
SHARK TANK INDIA - EDA INSIGHTS REPORT
{'='*80}

📊 DATASET OVERVIEW:
   • Total Pitches: {summary['total_rows']:,}
   • Total Features: {summary['total_columns']}
   • Numerical Variables: {summary['numerical_vars']}
   • Categorical Variables: {summary['categorical_vars']}
   • Missing Values: {summary['missing_values']:,}
   • Duplicate Rows: {summary['duplicate_rows']}

📈 DISTRIBUTION ANALYSIS:
   • Right-skewed variables: {len([col for col in self.numerical_cols if skew(self.df[col].dropna()) > 0.5])}
   • Left-skewed variables: {len([col for col in self.numerical_cols if skew(self.df[col].dropna()) < -0.5])}
   • Symmetric variables: {len([col for col in self.numerical_cols if abs(skew(self.df[col].dropna())) <= 0.5])}

🔍 DATA QUALITY:
   • Columns with >50% missing: {len(self.missing_data_analysis()[self.missing_data_analysis()['Missing_Percentage'] > 50])}
   • Columns with >20% missing: {len(self.missing_data_analysis()[self.missing_data_analysis()['Missing_Percentage'] > 20])}

💡 RECOMMENDATIONS:
   1. Handle missing data using appropriate imputation strategies
   2. Consider log transformation for right-skewed variables
   3. Investigate outliers in key financial metrics
   4. Create categorical bins for continuous variables
   5. Engineer interaction features for modeling

{'='*80}
"""
        return report


class VisualizationEngine:
    """Advanced visualization utilities for EDA"""
    
    @staticmethod
    def plot_distribution_grid(df: pd.DataFrame, columns: List[str], 
                               figsize: Tuple[int, int] = (16, 12),
                               save_path: Optional[str] = None):
        """Create grid of distribution plots"""
        n_cols = 2
        n_rows = (len(columns) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.ravel()
        
        for idx, col in enumerate(columns):
            data = df[col].dropna()
            
            axes[idx].hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                            label=f'Mean: {data.mean():.2f}')
            axes[idx].axvline(data.median(), color='green', linestyle='--', linewidth=2,
                            label=f'Median: {data.median():.2f}')
            axes[idx].set_title(f'{col.replace("_", " ").title()}', fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, columns: Optional[List[str]] = None,
                                figsize: Tuple[int, int] = (14, 12),
                                save_path: Optional[str] = None):
        """Create correlation heatmap"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr_matrix = df[columns].corr()
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    @staticmethod
    def plot_categorical_bars(df: pd.DataFrame, column: str, top_n: int = 15,
                             figsize: Tuple[int, int] = (14, 8),
                             save_path: Optional[str] = None):
        """Create horizontal bar chart for categorical variable"""
        counts = df[column].value_counts().head(top_n)
        
        plt.figure(figsize=figsize)
        colors = plt.cm.Set3(range(len(counts)))
        plt.barh(counts.index, counts.values, color=colors, edgecolor='black')
        plt.xlabel('Count', fontsize=12)
        plt.title(f'Top {top_n} {column.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, axis='x')
        
        for i, v in enumerate(counts.values):
            plt.text(v + max(counts.values)*0.01, i, str(v), va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
