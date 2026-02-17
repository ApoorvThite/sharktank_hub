"""
DEAL STRUCTURE RECOMMENDATION ENGINE
Recommend optimal deal structure based on startup profile and historical patterns
"""

import pandas as pd
import numpy as np
import pickle
import json

class DealStructureRecommender:
    """
    Recommend optimal deal structure (Pure Equity, Debt+Equity, Royalty+Equity)
    based on startup characteristics and historical Shark Tank India patterns
    """
    
    def __init__(self, 
                 model_path='models/deal_structure_predictor.pkl',
                 data_path='data/processed/processed_data_with_deal_structures.csv'):
        """Initialize recommender with trained model and historical data"""
        
        print("🔧 Initializing Deal Structure Recommender...")
        
        # Load predictive model
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
            print(f"✅ Loaded predictive model")
        except:
            self.model = None
            self.feature_names = None
            print(f"⚠️  Model not found, using rule-based system only")
        
        # Load historical data for benchmarking
        try:
            self.df = pd.read_csv(data_path)
            print(f"✅ Loaded historical data ({len(self.df)} startups)")
        except:
            self.df = None
            print(f"⚠️  Historical data not found")
    
    def recommend_deal_structure(self, startup_profile):
        """
        Recommend optimal deal structure based on startup profile
        
        Parameters:
        -----------
        startup_profile : dict
            Startup characteristics:
            - yearly_revenue: float (in lakhs)
            - gross_margin: float (percentage)
            - net_margin: float (percentage)
            - cash_burn: bool (True if burning cash)
            - industry: str
            - ask_amount: float (in lakhs)
            - valuation: float (in lakhs)
        
        Returns:
        --------
        dict : Recommendation with structure, reasoning, and alternatives
        """
        
        revenue = startup_profile.get('yearly_revenue', 0)
        gross_margin = startup_profile.get('gross_margin', 0)
        net_margin = startup_profile.get('net_margin', 0)
        cash_burn = startup_profile.get('cash_burn', False)
        industry = startup_profile.get('industry', 'Unknown')
        ask_amount = startup_profile.get('ask_amount', 50)
        valuation = startup_profile.get('valuation', 2000)
        
        # Default: pure equity
        recommended = 'Pure Equity'
        reasoning = []
        confidence = 0.5
        
        # Rule-based recommendation system
        
        # RULE 1: Debt component if stable revenue + positive cash flow + asset-heavy
        debt_score = 0
        if revenue > 200:
            debt_score += 0.3
            reasoning.append(f"Revenue ₹{revenue:.0f}L supports debt servicing")
        
        if not cash_burn:
            debt_score += 0.3
            reasoning.append("Positive cash flow enables debt repayment")
        
        if industry in ['Manufacturing', 'Food and Beverage', 'Vehicles/Electrical Vehicles']:
            debt_score += 0.2
            reasoning.append(f"{industry} is asset-heavy, suitable for debt")
        
        if gross_margin < 50:
            debt_score += 0.2
            reasoning.append("Lower margins suggest debt over dilution")
        
        # RULE 2: Royalty if high margins + B2C + moderate revenue
        royalty_score = 0
        if gross_margin > 60:
            royalty_score += 0.4
            reasoning.append(f"High gross margin ({gross_margin:.1f}%) can absorb royalty")
        
        if industry in ['Beauty/Fashion', 'Food and Beverage', 'Lifestyle/Home']:
            royalty_score += 0.3
            reasoning.append(f"{industry} is B2C with recurring revenue")
        
        if 300 < revenue < 1000:
            royalty_score += 0.2
            reasoning.append("Moderate revenue suitable for royalty structure")
        
        if net_margin > 20:
            royalty_score += 0.1
            reasoning.append("Strong net margins support royalty payments")
        
        # RULE 3: Pure equity if early stage or high growth
        equity_score = 0.5  # Default baseline
        
        if revenue < 100:
            equity_score += 0.3
            reasoning.append("Early-stage revenue best suited for pure equity")
        
        if industry in ['Technology/Software', 'Medical/Health']:
            equity_score += 0.2
            reasoning.append(f"{industry} typically uses pure equity for growth")
        
        if gross_margin > 70:
            equity_score += 0.1
            reasoning.append("Exceptional margins indicate high-growth potential")
        
        if cash_burn:
            equity_score += 0.2
            reasoning.append("Cash burn phase requires equity, not debt burden")
        
        # Determine recommendation based on highest score
        scores = {
            'Pure Equity': equity_score,
            'Debt + Equity': debt_score,
            'Royalty + Equity': royalty_score
        }
        
        recommended = max(scores, key=scores.get)
        confidence = scores[recommended]
        
        # Additional context from historical data
        historical_context = self._get_historical_context(startup_profile)
        
        # Build recommendation
        recommendation = {
            'recommended_structure': recommended,
            'confidence': min(confidence, 1.0),
            'reasoning': reasoning[:5],  # Top 5 reasons
            'alternatives': self._rank_alternatives(scores, recommended),
            'historical_context': historical_context,
            'typical_terms': self._get_typical_terms(recommended, startup_profile)
        }
        
        return recommendation
    
    def _get_historical_context(self, startup_profile):
        """Get historical context for similar startups"""
        if self.df is None:
            return {}
        
        industry = startup_profile.get('industry', 'Unknown')
        revenue = startup_profile.get('yearly_revenue', 0)
        
        # Filter similar startups
        similar = self.df[
            (self.df['Industry'] == industry) & 
            (self.df['Received Offer'] == 1)
        ].copy()
        
        if len(similar) == 0:
            return {'message': 'No historical data for this industry'}
        
        # Calculate statistics
        structure_dist = similar['deal_structure'].value_counts()
        
        context = {
            'similar_deals': len(similar),
            'most_common_structure': structure_dist.index[0] if len(structure_dist) > 0 else 'Unknown',
            'structure_distribution': structure_dist.to_dict(),
            'avg_deal_size': float(similar['Total Deal Amount'].mean()),
            'avg_equity': float(similar['Total Deal Equity'].mean())
        }
        
        return context
    
    def _rank_alternatives(self, scores, recommended):
        """Rank alternative structures"""
        alternatives = []
        for structure, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if structure != recommended:
                alternatives.append({
                    'structure': structure,
                    'score': float(score),
                    'viability': 'High' if score > 0.6 else 'Medium' if score > 0.4 else 'Low'
                })
        return alternatives
    
    def _get_typical_terms(self, structure, startup_profile):
        """Get typical terms for recommended structure"""
        if self.df is None:
            return {}
        
        industry = startup_profile.get('industry', 'Unknown')
        revenue = startup_profile.get('yearly_revenue', 0)
        
        # Filter by structure
        structure_deals = self.df[self.df['deal_structure'] == structure].copy()
        
        if len(structure_deals) == 0:
            return {'message': 'No historical data for this structure'}
        
        terms = {
            'avg_deal_amount': float(structure_deals['Total Deal Amount'].mean()),
            'avg_equity': float(structure_deals['Total Deal Equity'].mean()),
        }
        
        # Add structure-specific terms
        if 'Debt' in structure:
            debt_deals = structure_deals[structure_deals['Total Deal Debt'] > 0]
            if len(debt_deals) > 0:
                terms['avg_debt_amount'] = float(debt_deals['Total Deal Debt'].mean())
                terms['avg_debt_interest'] = float(debt_deals['Debt Interest'].mean())
                terms['avg_debt_ratio'] = float(
                    (debt_deals['Total Deal Debt'] / 
                     (debt_deals['Total Deal Amount'] + debt_deals['Total Deal Debt'])).mean()
                )
        
        if 'Royalty' in structure:
            royalty_deals = structure_deals[structure_deals['Royalty Percentage'] > 0]
            if len(royalty_deals) > 0:
                terms['avg_royalty_pct'] = float(royalty_deals['Royalty Percentage'].mean())
                if 'Royalty Recouped Amount' in royalty_deals.columns:
                    recoup = royalty_deals[royalty_deals['Royalty Recouped Amount'] > 0]
                    if len(recoup) > 0:
                        terms['avg_recoupment'] = float(recoup['Royalty Recouped Amount'].mean())
        
        return terms
    
    def explain_recommendation(self, recommendation, startup_profile):
        """Provide detailed explanation of recommendation"""
        
        explanation = f"\n{'='*70}\n"
        explanation += f"💼 DEAL STRUCTURE RECOMMENDATION\n"
        explanation += f"{'='*70}\n\n"
        
        # Startup profile
        explanation += f"📊 Startup Profile:\n"
        explanation += f"   Industry: {startup_profile.get('industry', 'Unknown')}\n"
        explanation += f"   Revenue: ₹{startup_profile.get('yearly_revenue', 0):.0f}L\n"
        explanation += f"   Gross Margin: {startup_profile.get('gross_margin', 0):.1f}%\n"
        explanation += f"   Net Margin: {startup_profile.get('net_margin', 0):.1f}%\n"
        explanation += f"   Cash Burn: {'Yes' if startup_profile.get('cash_burn', False) else 'No'}\n"
        
        # Recommendation
        explanation += f"\n🎯 Recommended Structure: {recommendation['recommended_structure']}\n"
        explanation += f"   Confidence: {recommendation['confidence']:.1%}\n"
        
        # Reasoning
        explanation += f"\n💡 Why This Structure:\n"
        for i, reason in enumerate(recommendation['reasoning'], 1):
            explanation += f"   {i}. {reason}\n"
        
        # Historical context
        if 'historical_context' in recommendation and 'similar_deals' in recommendation['historical_context']:
            ctx = recommendation['historical_context']
            explanation += f"\n📈 Historical Context:\n"
            explanation += f"   Similar deals in {startup_profile.get('industry', 'this industry')}: {ctx['similar_deals']}\n"
            explanation += f"   Most common structure: {ctx['most_common_structure']}\n"
            explanation += f"   Avg deal size: ₹{ctx['avg_deal_size']:.1f}L\n"
            explanation += f"   Avg equity: {ctx['avg_equity']:.1f}%\n"
        
        # Typical terms
        if 'typical_terms' in recommendation and 'avg_deal_amount' in recommendation['typical_terms']:
            terms = recommendation['typical_terms']
            explanation += f"\n💰 Typical Terms for {recommendation['recommended_structure']}:\n"
            explanation += f"   Deal Amount: ₹{terms['avg_deal_amount']:.1f}L\n"
            explanation += f"   Equity: {terms['avg_equity']:.1f}%\n"
            
            if 'avg_debt_amount' in terms:
                explanation += f"   Debt Amount: ₹{terms['avg_debt_amount']:.1f}L\n"
                explanation += f"   Debt Interest: {terms['avg_debt_interest']:.1f}%\n"
                explanation += f"   Debt Ratio: {terms['avg_debt_ratio']:.1%}\n"
            
            if 'avg_royalty_pct' in terms:
                explanation += f"   Royalty: {terms['avg_royalty_pct']:.2f}%\n"
                if 'avg_recoupment' in terms:
                    explanation += f"   Recoupment: ₹{terms['avg_recoupment']:.0f}L\n"
        
        # Alternatives
        explanation += f"\n🔄 Alternative Structures:\n"
        for alt in recommendation['alternatives']:
            explanation += f"   • {alt['structure']}: {alt['viability']} viability (score: {alt['score']:.2f})\n"
        
        explanation += f"\n{'='*70}\n"
        
        return explanation


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("💼 DEAL STRUCTURE RECOMMENDATION ENGINE - DEMO")
    print("="*80)
    
    # Initialize recommender
    recommender = DealStructureRecommender()
    
    # Example 1: High-revenue F&B startup (debt candidate)
    print("\n" + "="*80)
    print("EXAMPLE 1: High-Revenue Food & Beverage Startup")
    print("="*80)
    
    startup1 = {
        'yearly_revenue': 450,
        'gross_margin': 52,
        'net_margin': 18,
        'cash_burn': False,
        'industry': 'Food and Beverage',
        'ask_amount': 75,
        'valuation': 3000
    }
    
    rec1 = recommender.recommend_deal_structure(startup1)
    print(recommender.explain_recommendation(rec1, startup1))
    
    # Example 2: High-margin Beauty startup (royalty candidate)
    print("\n" + "="*80)
    print("EXAMPLE 2: High-Margin Beauty/Fashion Startup")
    print("="*80)
    
    startup2 = {
        'yearly_revenue': 600,
        'gross_margin': 68,
        'net_margin': 24,
        'cash_burn': False,
        'industry': 'Beauty/Fashion',
        'ask_amount': 80,
        'valuation': 4500
    }
    
    rec2 = recommender.recommend_deal_structure(startup2)
    print(recommender.explain_recommendation(rec2, startup2))
    
    # Example 3: Early-stage Tech startup (pure equity)
    print("\n" + "="*80)
    print("EXAMPLE 3: Early-Stage Technology Startup")
    print("="*80)
    
    startup3 = {
        'yearly_revenue': 80,
        'gross_margin': 75,
        'net_margin': 35,
        'cash_burn': True,
        'industry': 'Technology/Software',
        'ask_amount': 60,
        'valuation': 2500
    }
    
    rec3 = recommender.recommend_deal_structure(startup3)
    print(recommender.explain_recommendation(rec3, startup3))
    
    # Example 4: Manufacturing startup (debt candidate)
    print("\n" + "="*80)
    print("EXAMPLE 4: Manufacturing Startup")
    print("="*80)
    
    startup4 = {
        'yearly_revenue': 350,
        'gross_margin': 45,
        'net_margin': 15,
        'cash_burn': False,
        'industry': 'Manufacturing',
        'ask_amount': 70,
        'valuation': 2000
    }
    
    rec4 = recommender.recommend_deal_structure(startup4)
    print(recommender.explain_recommendation(rec4, startup4))
    
    print("\n" + "="*80)
    print("✅ DEAL STRUCTURE RECOMMENDER DEMO COMPLETE")
    print("="*80)
    print("\nThis tool can be used to:")
    print("   • Recommend optimal deal structures for any startup")
    print("   • Provide reasoning based on historical patterns")
    print("   • Suggest typical terms for each structure")
    print("   • Compare alternatives with viability scores")
    print("\nBased on analysis of 465 Shark Tank India deals.")
    print("="*80)
