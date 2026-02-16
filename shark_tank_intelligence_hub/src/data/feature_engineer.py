import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
        
    def create_features(self, df):
        df_features = df.copy()
        
        df_features = self._create_deal_features(df_features)
        df_features = self._create_valuation_features(df_features)
        df_features = self._create_shark_features(df_features)
        df_features = self._create_temporal_features(df_features)
        df_features = self._create_industry_features(df_features)
        df_features = self._create_geographic_features(df_features)
        
        return df_features
    
    def _create_deal_features(self, df):
        if 'deal_accepted' in df.columns:
            df['deal_success'] = df['deal_accepted'].astype(int)
        
        if 'amount_invested' in df.columns and 'equity_taken' in df.columns:
            df['implied_valuation'] = np.where(
                df['equity_taken'] > 0,
                (df['amount_invested'] / df['equity_taken']) * 100,
                np.nan
            )
            self.feature_names.append('implied_valuation')
        
        return df
    
    def _create_valuation_features(self, df):
        if 'asked_amount' in df.columns and 'asked_equity' in df.columns:
            df['asked_valuation'] = np.where(
                df['asked_equity'] > 0,
                (df['asked_amount'] / df['asked_equity']) * 100,
                np.nan
            )
            self.feature_names.append('asked_valuation')
        
        if 'implied_valuation' in df.columns and 'asked_valuation' in df.columns:
            df['valuation_gap'] = df['asked_valuation'] - df['implied_valuation']
            df['valuation_gap_pct'] = (df['valuation_gap'] / df['asked_valuation']) * 100
            self.feature_names.extend(['valuation_gap', 'valuation_gap_pct'])
        
        return df
    
    def _create_shark_features(self, df):
        shark_cols = [col for col in df.columns if 'shark' in col.lower()]
        if shark_cols:
            df['num_sharks_involved'] = df[shark_cols].sum(axis=1)
            self.feature_names.append('num_sharks_involved')
        
        return df
    
    def _create_temporal_features(self, df):
        if 'season' in df.columns:
            df['season_num'] = df['season'].astype(int)
            self.feature_names.append('season_num')
        
        if 'episode' in df.columns:
            df['episode_num'] = df['episode'].astype(int)
            self.feature_names.append('episode_num')
        
        return df
    
    def _create_industry_features(self, df):
        if 'industry' in df.columns:
            df['industry_encoded'] = pd.Categorical(df['industry']).codes
            self.feature_names.append('industry_encoded')
        
        return df
    
    def _create_geographic_features(self, df):
        if 'location' in df.columns or 'state' in df.columns:
            location_col = 'state' if 'state' in df.columns else 'location'
            df['location_encoded'] = pd.Categorical(df[location_col]).codes
            self.feature_names.append('location_encoded')
        
        return df
    
    def get_feature_list(self):
        return self.feature_names
