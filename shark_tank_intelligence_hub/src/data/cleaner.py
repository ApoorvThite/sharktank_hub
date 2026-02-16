import pandas as pd
import numpy as np
import re

class DataCleaner:
    def __init__(self):
        self.cleaning_log = []
        
    def clean_dataset(self, df):
        df_clean = df.copy()
        
        df_clean = self._handle_missing_values(df_clean)
        df_clean = self._standardize_columns(df_clean)
        df_clean = self._clean_numeric_fields(df_clean)
        df_clean = self._clean_categorical_fields(df_clean)
        df_clean = self._remove_duplicates(df_clean)
        
        return df_clean
    
    def _handle_missing_values(self, df):
        self.cleaning_log.append(f"Missing values before: {df.isnull().sum().sum()}")
        return df
    
    def _standardize_columns(self, df):
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        return df
    
    def _clean_numeric_fields(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def _clean_categorical_fields(self, df):
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]
        return df
    
    def _remove_duplicates(self, df):
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        if removed > 0:
            self.cleaning_log.append(f"Removed {removed} duplicate rows")
        return df
    
    def get_cleaning_report(self):
        return "\n".join(self.cleaning_log)
