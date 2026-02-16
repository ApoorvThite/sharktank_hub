import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_project_root():
    return Path(__file__).parent.parent.parent

def calculate_basic_stats(df):
    stats = {
        'total_pitches': len(df),
        'total_deals': df['deal_accepted'].sum() if 'deal_accepted' in df.columns else 0,
        'success_rate': (df['deal_accepted'].sum() / len(df) * 100) if 'deal_accepted' in df.columns else 0,
        'total_investment': df['amount_invested'].sum() if 'amount_invested' in df.columns else 0,
        'avg_deal_size': df['amount_invested'].mean() if 'amount_invested' in df.columns else 0,
        'avg_equity': df['equity_taken'].mean() if 'equity_taken' in df.columns else 0
    }
    return stats

def format_currency(amount):
    if amount >= 10000000:
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:
        return f"₹{amount/100000:.2f} L"
    else:
        return f"₹{amount:,.0f}"

def get_shark_columns(df):
    shark_keywords = ['aman', 'anupam', 'namita', 'vineeta', 'peyush', 'ashneer', 'ghazal']
    shark_cols = []
    
    for col in df.columns:
        if any(shark in col.lower() for shark in shark_keywords):
            shark_cols.append(col)
    
    return shark_cols

def filter_dataframe(df, filters):
    filtered_df = df.copy()
    
    if 'seasons' in filters and filters['seasons']:
        filtered_df = filtered_df[filtered_df['season'].isin(filters['seasons'])]
    
    if 'industries' in filters and filters['industries']:
        filtered_df = filtered_df[filtered_df['industry'].isin(filters['industries'])]
    
    if 'states' in filters and filters['states']:
        filtered_df = filtered_df[filtered_df['state'].isin(filters['states'])]
    
    return filtered_df

def create_summary_stats(df):
    summary = {
        'Total Rows': len(df),
        'Total Columns': len(df.columns),
        'Missing Values': df.isnull().sum().sum(),
        'Duplicate Rows': df.duplicated().sum()
    }
    return summary

@st.cache_data
def load_cached_data(file_path):
    return pd.read_csv(file_path)

def validate_dataset(df):
    required_columns = ['startup_name', 'industry', 'season']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    return True, "Dataset validated successfully"
