import pandas as pd
import os
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
    def load_raw_data(self, filename='shark_tank_india.csv'):
        filepath = self.raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Raw data file not found: {filepath}")
        return pd.read_csv(filepath)
    
    def load_processed_data(self, filename='processed_data.csv'):
        filepath = self.processed_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Processed data file not found: {filepath}")
        return pd.read_csv(filepath)
    
    def save_processed_data(self, df, filename='processed_data.csv'):
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.processed_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved processed data to {filepath}")
        
    def load_industry_benchmarks(self):
        filepath = self.data_dir / 'industry_benchmarks' / 'benchmarks.csv'
        if filepath.exists():
            return pd.read_csv(filepath)
        return None
    
    def load_shark_profiles(self):
        filepath = self.data_dir / 'shark_profiles' / 'profiles.csv'
        if filepath.exists():
            return pd.read_csv(filepath)
        return None
