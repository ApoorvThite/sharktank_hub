import pandas as pd
import numpy as np

class IndustryProfiler:
    def __init__(self):
        self.industry_stats = {}
        
    def analyze_industries(self, df, industry_col='industry'):
        industries = df[industry_col].unique()
        
        for industry in industries:
            industry_data = df[df[industry_col] == industry]
            
            self.industry_stats[industry] = {
                'total_pitches': len(industry_data),
                'deals_made': industry_data.get('deal_accepted', pd.Series([0])).sum(),
                'success_rate': (industry_data.get('deal_accepted', pd.Series([0])).sum() / len(industry_data) * 100) if len(industry_data) > 0 else 0,
                'avg_investment': industry_data.get('amount_invested', pd.Series([0])).mean(),
                'avg_equity': industry_data.get('equity_taken', pd.Series([0])).mean(),
                'total_investment': industry_data.get('amount_invested', pd.Series([0])).sum(),
                'avg_valuation': industry_data.get('implied_valuation', pd.Series([0])).mean()
            }
        
        return self.industry_stats
    
    def get_industry_benchmarks(self, industry):
        if industry not in self.industry_stats:
            return None
        
        return self.industry_stats[industry]
    
    def get_top_industries(self, metric='total_pitches', top_n=10):
        sorted_industries = sorted(
            self.industry_stats.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True
        )
        
        return sorted_industries[:top_n]
    
    def compare_industries(self, industry1, industry2):
        if industry1 not in self.industry_stats or industry2 not in self.industry_stats:
            return None
        
        comparison = {}
        for metric in self.industry_stats[industry1].keys():
            comparison[metric] = {
                industry1: self.industry_stats[industry1][metric],
                industry2: self.industry_stats[industry2][metric],
                'difference': self.industry_stats[industry1][metric] - self.industry_stats[industry2][metric]
            }
        
        return comparison
    
    def get_industry_trends(self, df, industry_col='industry', season_col='season'):
        trends = {}
        
        for industry in df[industry_col].unique():
            industry_data = df[df[industry_col] == industry]
            
            season_trends = industry_data.groupby(season_col).agg({
                'deal_accepted': 'sum',
                'amount_invested': 'mean',
                'equity_taken': 'mean'
            }).to_dict('index')
            
            trends[industry] = season_trends
        
        return trends
    
    def generate_industry_report(self):
        report_df = pd.DataFrame.from_dict(self.industry_stats, orient='index')
        report_df = report_df.sort_values('total_pitches', ascending=False)
        
        return report_df
