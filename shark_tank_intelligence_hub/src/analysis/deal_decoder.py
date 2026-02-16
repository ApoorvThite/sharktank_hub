import pandas as pd
import numpy as np

class DealDecoder:
    def __init__(self):
        self.deal_patterns = {}
        
    def analyze_deal_structures(self, df):
        deal_types = {
            'equity_only': 0,
            'equity_with_debt': 0,
            'equity_with_royalty': 0,
            'complex_deals': 0
        }
        
        for _, row in df.iterrows():
            has_debt = row.get('debt_amount', 0) > 0
            has_royalty = row.get('royalty', 0) > 0
            
            if has_debt and has_royalty:
                deal_types['complex_deals'] += 1
            elif has_debt:
                deal_types['equity_with_debt'] += 1
            elif has_royalty:
                deal_types['equity_with_royalty'] += 1
            else:
                deal_types['equity_only'] += 1
        
        self.deal_patterns['structure_distribution'] = deal_types
        return deal_types
    
    def analyze_debt_terms(self, df):
        debt_deals = df[df.get('debt_amount', pd.Series([0])) > 0]
        
        if len(debt_deals) == 0:
            return None
        
        debt_analysis = {
            'total_debt_deals': len(debt_deals),
            'avg_debt_amount': debt_deals.get('debt_amount', pd.Series([0])).mean(),
            'avg_interest_rate': debt_deals.get('interest_rate', pd.Series([0])).mean(),
            'total_debt_issued': debt_deals.get('debt_amount', pd.Series([0])).sum()
        }
        
        return debt_analysis
    
    def analyze_royalty_terms(self, df):
        royalty_deals = df[df.get('royalty', pd.Series([0])) > 0]
        
        if len(royalty_deals) == 0:
            return None
        
        royalty_analysis = {
            'total_royalty_deals': len(royalty_deals),
            'avg_royalty_percentage': royalty_deals.get('royalty', pd.Series([0])).mean(),
            'avg_royalty_duration': royalty_deals.get('royalty_duration', pd.Series([0])).mean()
        }
        
        return royalty_analysis
    
    def identify_special_terms(self, df):
        special_terms = []
        
        for idx, row in df.iterrows():
            terms = []
            
            if row.get('debt_amount', 0) > 0:
                terms.append(f"Debt: ₹{row.get('debt_amount', 0)}L")
            
            if row.get('royalty', 0) > 0:
                terms.append(f"Royalty: {row.get('royalty', 0)}%")
            
            if row.get('advisory_equity', 0) > 0:
                terms.append(f"Advisory: {row.get('advisory_equity', 0)}%")
            
            if terms:
                special_terms.append({
                    'pitch_id': idx,
                    'startup': row.get('startup_name', 'Unknown'),
                    'special_terms': ', '.join(terms)
                })
        
        return special_terms
    
    def calculate_effective_cost(self, amount, equity, debt=0, royalty=0, royalty_duration=0):
        equity_cost = amount / (equity / 100) if equity > 0 else 0
        
        total_debt_cost = debt * (1 + 0.12)
        
        total_royalty = (royalty / 100) * royalty_duration * amount if royalty > 0 else 0
        
        effective_cost = equity_cost + total_debt_cost + total_royalty
        
        return {
            'equity_cost': equity_cost,
            'debt_cost': total_debt_cost,
            'royalty_cost': total_royalty,
            'total_effective_cost': effective_cost
        }
    
    def get_deal_complexity_score(self, row):
        score = 0
        
        if row.get('equity_taken', 0) > 0:
            score += 1
        
        if row.get('debt_amount', 0) > 0:
            score += 2
        
        if row.get('royalty', 0) > 0:
            score += 2
        
        if row.get('advisory_equity', 0) > 0:
            score += 1
        
        num_sharks = sum([1 for col in row.index if 'shark' in col.lower() and row[col] == 1])
        score += num_sharks
        
        return score
