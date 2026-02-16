import pandas as pd
import numpy as np

class GeoMapper:
    def __init__(self):
        self.location_stats = {}
        
    def analyze_geographic_patterns(self, df, location_col='state'):
        locations = df[location_col].unique()
        
        for location in locations:
            location_data = df[df[location_col] == location]
            
            self.location_stats[location] = {
                'total_pitches': len(location_data),
                'deals_made': location_data.get('deal_accepted', pd.Series([0])).sum(),
                'success_rate': (location_data.get('deal_accepted', pd.Series([0])).sum() / len(location_data) * 100) if len(location_data) > 0 else 0,
                'avg_investment': location_data.get('amount_invested', pd.Series([0])).mean(),
                'total_investment': location_data.get('amount_invested', pd.Series([0])).sum(),
                'avg_equity': location_data.get('equity_taken', pd.Series([0])).mean()
            }
        
        return self.location_stats
    
    def get_top_locations(self, metric='total_pitches', top_n=10):
        sorted_locations = sorted(
            self.location_stats.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True
        )
        
        return sorted_locations[:top_n]
    
    def get_location_benchmarks(self, location):
        if location not in self.location_stats:
            return None
        
        return self.location_stats[location]
    
    def analyze_regional_trends(self, df, location_col='state', season_col='season'):
        trends = {}
        
        for location in df[location_col].unique():
            location_data = df[df[location_col] == location]
            
            season_trends = location_data.groupby(season_col).agg({
                'deal_accepted': 'sum',
                'amount_invested': 'sum'
            }).to_dict('index')
            
            trends[location] = season_trends
        
        return trends
    
    def identify_startup_hubs(self, min_pitches=10):
        hubs = []
        
        for location, stats in self.location_stats.items():
            if stats['total_pitches'] >= min_pitches:
                hubs.append({
                    'location': location,
                    'total_pitches': stats['total_pitches'],
                    'success_rate': stats['success_rate'],
                    'total_investment': stats['total_investment']
                })
        
        return sorted(hubs, key=lambda x: x['total_pitches'], reverse=True)
    
    def compare_locations(self, location1, location2):
        if location1 not in self.location_stats or location2 not in self.location_stats:
            return None
        
        comparison = {}
        for metric in self.location_stats[location1].keys():
            comparison[metric] = {
                location1: self.location_stats[location1][metric],
                location2: self.location_stats[location2][metric],
                'difference': self.location_stats[location1][metric] - self.location_stats[location2][metric]
            }
        
        return comparison
    
    def generate_geographic_report(self):
        report_df = pd.DataFrame.from_dict(self.location_stats, orient='index')
        report_df = report_df.sort_values('total_pitches', ascending=False)
        
        return report_df
