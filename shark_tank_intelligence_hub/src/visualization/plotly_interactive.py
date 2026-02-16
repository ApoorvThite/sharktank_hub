import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class PlotlyVisualizer:
    def __init__(self):
        self.color_scheme = px.colors.qualitative.Set2
        
    def plot_success_rate_by_industry(self, df, industry_col='industry'):
        industry_stats = df.groupby(industry_col).agg({
            'deal_accepted': ['sum', 'count']
        }).reset_index()
        
        industry_stats.columns = [industry_col, 'deals', 'total']
        industry_stats['success_rate'] = (industry_stats['deals'] / industry_stats['total'] * 100)
        industry_stats = industry_stats.sort_values('success_rate', ascending=False)
        
        fig = px.bar(
            industry_stats,
            x='success_rate',
            y=industry_col,
            orientation='h',
            title='Success Rate by Industry',
            labels={'success_rate': 'Success Rate (%)', industry_col: 'Industry'},
            color='success_rate',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=600, showlegend=False)
        return fig
    
    def plot_investment_trends(self, df, season_col='season'):
        season_stats = df.groupby(season_col).agg({
            'amount_invested': 'sum',
            'deal_accepted': 'sum'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=season_stats[season_col], y=season_stats['amount_invested'], 
                   name='Total Investment (₹L)'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=season_stats[season_col], y=season_stats['deal_accepted'],
                      name='Number of Deals', mode='lines+markers'),
            secondary_y=True
        )
        
        fig.update_layout(title='Investment Trends Across Seasons')
        fig.update_xaxes(title_text='Season')
        fig.update_yaxes(title_text='Investment (₹L)', secondary_y=False)
        fig.update_yaxes(title_text='Number of Deals', secondary_y=True)
        
        return fig
    
    def plot_shark_performance(self, shark_stats_df):
        fig = px.scatter(
            shark_stats_df,
            x='total_deals',
            y='total_investment',
            size='avg_equity',
            color='shark_name',
            hover_data=['success_rate'],
            title='Shark Performance Overview',
            labels={
                'total_deals': 'Total Deals',
                'total_investment': 'Total Investment (₹L)',
                'avg_equity': 'Avg Equity (%)'
            }
        )
        
        fig.update_layout(height=600)
        return fig
    
    def plot_valuation_distribution(self, df, valuation_col='implied_valuation'):
        fig = px.histogram(
            df,
            x=valuation_col,
            nbins=50,
            title='Distribution of Startup Valuations',
            labels={valuation_col: 'Valuation (₹L)'},
            color_discrete_sequence=['#636EFA']
        )
        
        fig.update_layout(showlegend=False)
        return fig
    
    def plot_geographic_heatmap(self, location_stats_df):
        fig = px.choropleth(
            location_stats_df,
            locations='state',
            locationmode='country names',
            color='total_investment',
            hover_data=['total_pitches', 'success_rate'],
            title='Geographic Distribution of Investments',
            color_continuous_scale='Reds'
        )
        
        return fig
    
    def plot_deal_structure_pie(self, deal_types):
        fig = px.pie(
            values=list(deal_types.values()),
            names=list(deal_types.keys()),
            title='Deal Structure Distribution',
            hole=0.3
        )
        
        return fig
    
    def create_interactive_dashboard(self, df):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Success Rate by Season', 'Investment by Industry',
                          'Equity Distribution', 'Deal Count by Shark'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'histogram'}, {'type': 'bar'}]]
        )
        
        return fig
