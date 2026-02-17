"""
VALUATION CALCULATOR - PRODUCTION TOOL
Standalone calculator for startup valuation assessment based on Shark Tank India data
"""

import pandas as pd
import numpy as np
from pathlib import Path

class ValuationCalculator:
    """
    Calculate recommended valuations and assess startup asks
    Based on Shark Tank India industry benchmarks
    """
    
    def __init__(self, benchmarks_path='reports/industry_benchmarks.csv'):
        """Load industry benchmarks"""
        try:
            self.benchmarks = pd.read_csv(benchmarks_path, index_col=0)
            print("✅ Loaded industry benchmarks")
        except FileNotFoundError:
            print("⚠️  Benchmarks file not found. Using default values.")
            self.benchmarks = None
    
    def calculate_recommended_valuation(
        self,
        industry,
        yearly_revenue,
        gross_margin=50,
        net_margin=20,
        has_patent=False,
        team_quality=5,  # 1-10 scale
        verbose=True
    ):
        """
        Calculate recommended valuation range for revenue-generating startup
        
        Parameters:
        -----------
        industry : str
            Industry category (e.g., 'Technology/Software', 'Food and Beverage')
        yearly_revenue : float
            Yearly revenue in lakhs (₹L)
        gross_margin : float
            Gross margin percentage (0-100)
        net_margin : float
            Net margin percentage (0-100)
        has_patent : bool
            Whether startup has patents/IP
        team_quality : int
            Team quality score 1-10 (IIT/IIM, past exits, etc.)
        verbose : bool
            Print detailed breakdown
        
        Returns:
        --------
        dict with recommended valuation, range, and adjustments
        """
        
        if yearly_revenue <= 0:
            if verbose:
                print("⚠️  Revenue is 0. Use calculate_pre_revenue_valuation() instead.")
            return None
        
        # Get base multiple from industry benchmarks
        if self.benchmarks is not None and industry in self.benchmarks.index:
            base_multiple = self.benchmarks.loc[industry, 'revenue_multiple_median']
        else:
            # Default multiples by industry type
            default_multiples = {
                'Technology/Software': 15.0,
                'Medical/Health': 12.0,
                'Beauty/Fashion': 8.0,
                'Food and Beverage': 7.0,
                'Manufacturing': 6.0,
                'Agriculture': 5.0,
                'Business Services': 8.0,
                'Lifestyle/Home': 7.0,
                'Children/Education': 9.0,
                'Green/CleanTech': 10.0,
                'Fitness/Sports/Outdoors': 7.0,
                'Liquor/Alcohol': 6.0,
                'Animal/Pets': 5.0,
                'Electronics': 8.0,
                'Vehicles/Electrical Vehicles': 10.0
            }
            base_multiple = default_multiples.get(industry, 8.0)
        
        # Base valuation from revenue
        base_valuation = yearly_revenue * base_multiple
        
        # Margin adjustments
        margin_multiplier = 1.0
        if gross_margin > 60:
            margin_multiplier += 0.2
        if net_margin > 25:
            margin_multiplier += 0.15
        
        # Patent/IP premium
        patent_multiplier = 1.4 if has_patent else 1.0
        
        # Team quality multiplier (0.8 to 1.2)
        team_multiplier = 0.8 + (team_quality / 25)
        
        # Calculate recommended valuation
        recommended_valuation = (
            base_valuation * 
            margin_multiplier * 
            patent_multiplier * 
            team_multiplier
        )
        
        # Range: ±20%
        lower_bound = recommended_valuation * 0.8
        upper_bound = recommended_valuation * 1.2
        
        result = {
            'recommended': round(recommended_valuation, 2),
            'range': (round(lower_bound, 2), round(upper_bound, 2)),
            'base_multiple': round(base_multiple, 2),
            'adjustments': {
                'margin': round(margin_multiplier, 2),
                'patent': round(patent_multiplier, 2),
                'team': round(team_multiplier, 2)
            }
        }
        
        if verbose:
            print("="*70)
            print("💰 VALUATION CALCULATION")
            print("="*70)
            print(f"\n📊 Inputs:")
            print(f"   Industry: {industry}")
            print(f"   Yearly Revenue: ₹{yearly_revenue:.2f}L")
            print(f"   Gross Margin: {gross_margin}%")
            print(f"   Net Margin: {net_margin}%")
            print(f"   Has Patent: {has_patent}")
            print(f"   Team Quality: {team_quality}/10")
            
            print(f"\n🔢 Calculation:")
            print(f"   Base Multiple: {base_multiple:.2f}x")
            print(f"   Base Valuation: ₹{base_valuation:.2f}L")
            print(f"   Margin Multiplier: {margin_multiplier:.2f}x")
            print(f"   Patent Multiplier: {patent_multiplier:.2f}x")
            print(f"   Team Multiplier: {team_multiplier:.2f}x")
            
            print(f"\n✅ Recommended Valuation: ₹{recommended_valuation:.2f}L")
            print(f"   Range: ₹{lower_bound:.2f}L - ₹{upper_bound:.2f}L")
            print("="*70)
        
        return result
    
    def assess_ask(self, industry, yearly_revenue, asked_valuation, verbose=True):
        """
        Assess if asked valuation is reasonable
        
        Parameters:
        -----------
        industry : str
            Industry category
        yearly_revenue : float
            Yearly revenue in lakhs
        asked_valuation : float
            Valuation being asked in lakhs
        verbose : bool
            Print assessment
        
        Returns:
        --------
        dict with inflation percentage, assessment, and recommended range
        """
        
        # Calculate recommended valuation
        calc = self.calculate_recommended_valuation(
            industry, yearly_revenue, 50, 20, verbose=False
        )
        
        if calc is None:
            return None
        
        # Calculate inflation
        inflation = (asked_valuation - calc['recommended']) / calc['recommended'] * 100
        
        # Assessment
        if inflation < -20:
            assessment = "Significantly undervalued - may raise concerns"
            color = "🟡"
        elif inflation < 0:
            assessment = "Below market - excellent for investors"
            color = "🟢"
        elif inflation < 20:
            assessment = "Fair market value"
            color = "🟢"
        elif inflation < 50:
            assessment = "Above market - needs strong justification"
            color = "🟠"
        else:
            assessment = "Significantly overvalued - unlikely to get deal"
            color = "🔴"
        
        result = {
            'inflation_pct': round(inflation, 2),
            'assessment': assessment,
            'recommended_range': calc['range'],
            'recommended': calc['recommended']
        }
        
        if verbose:
            print("="*70)
            print("📊 VALUATION ASSESSMENT")
            print("="*70)
            print(f"\n💼 Startup Details:")
            print(f"   Industry: {industry}")
            print(f"   Revenue: ₹{yearly_revenue:.2f}L")
            print(f"   Asked Valuation: ₹{asked_valuation:.2f}L")
            
            print(f"\n🎯 Assessment:")
            print(f"   Recommended: ₹{calc['recommended']:.2f}L")
            print(f"   Range: ₹{calc['range'][0]:.2f}L - ₹{calc['range'][1]:.2f}L")
            print(f"   Inflation: {inflation:+.1f}%")
            print(f"   {color} {assessment}")
            print("="*70)
        
        return result
    
    def calculate_pre_revenue_valuation(
        self,
        industry,
        team_pedigree,      # 1-10: IIT/IIM, past exits, etc.
        market_size,        # TAM in crores
        product_stage,      # 'idea', 'prototype', 'beta', 'launched'
        traction=0,         # users/waitlist/LOIs
        has_patent=False,
        verbose=True
    ):
        """
        Special valuation model for pre-revenue startups
        
        Parameters:
        -----------
        industry : str
            Industry category
        team_pedigree : int
            Team quality 1-10 (education, experience, exits)
        market_size : float
            Total addressable market in crores
        product_stage : str
            One of: 'idea', 'prototype', 'beta', 'launched'
        traction : int
            Number of users, waitlist, or LOIs
        has_patent : bool
            Whether startup has patents/IP
        verbose : bool
            Print detailed breakdown
        
        Returns:
        --------
        float - recommended pre-revenue valuation in lakhs
        """
        
        # Base valuations by industry (in lakhs)
        base_valuations = {
            'Technology/Software': 1000,      # ₹10Cr
            'Medical/Health': 1500,           # ₹15Cr
            'Food and Beverage': 300,         # ₹3Cr
            'Beauty/Fashion': 500,            # ₹5Cr
            'Manufacturing': 400,             # ₹4Cr
            'Agriculture': 350,               # ₹3.5Cr
            'Business Services': 600,         # ₹6Cr
            'Lifestyle/Home': 400,            # ₹4Cr
            'Children/Education': 500,        # ₹5Cr
            'Green/CleanTech': 800,           # ₹8Cr
            'Fitness/Sports/Outdoors': 400,   # ₹4Cr
            'Electronics': 700,               # ₹7Cr
            'Vehicles/Electrical Vehicles': 1200  # ₹12Cr
        }
        
        base = base_valuations.get(industry, 500)
        
        # Team multiplier (1.0 - 1.5x)
        team_mult = 1.0 + (team_pedigree / 20)
        
        # Market size multiplier
        if market_size > 10000:  # >₹100Cr TAM
            market_mult = 1.3
        elif market_size > 1000:  # >₹10Cr TAM
            market_mult = 1.15
        else:
            market_mult = 1.0
        
        # Stage multiplier
        stage_mult = {
            'idea': 0.7,
            'prototype': 0.85,
            'beta': 1.0,
            'launched': 1.2
        }.get(product_stage.lower(), 1.0)
        
        # Traction boost (max 1.3x)
        traction_boost = min(1.0 + (traction / 10000), 1.3)
        
        # Patent premium
        patent_mult = 1.4 if has_patent else 1.0
        
        # Calculate
        valuation = (
            base * 
            team_mult * 
            market_mult * 
            stage_mult * 
            traction_boost * 
            patent_mult
        )
        
        # Round to nearest 10L
        valuation = round(valuation / 10) * 10
        
        if verbose:
            print("="*70)
            print("🚀 PRE-REVENUE VALUATION")
            print("="*70)
            print(f"\n📊 Inputs:")
            print(f"   Industry: {industry}")
            print(f"   Team Pedigree: {team_pedigree}/10")
            print(f"   Market Size: ₹{market_size:.0f}Cr")
            print(f"   Product Stage: {product_stage}")
            print(f"   Traction: {traction:,} users/LOIs")
            print(f"   Has Patent: {has_patent}")
            
            print(f"\n🔢 Multipliers:")
            print(f"   Base Valuation: ₹{base:.0f}L")
            print(f"   Team: {team_mult:.2f}x")
            print(f"   Market: {market_mult:.2f}x")
            print(f"   Stage: {stage_mult:.2f}x")
            print(f"   Traction: {traction_boost:.2f}x")
            print(f"   Patent: {patent_mult:.2f}x")
            
            print(f"\n✅ Recommended Valuation: ₹{valuation:.0f}L (₹{valuation/100:.1f}Cr)")
            print("="*70)
        
        return valuation


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("💰 SHARK TANK VALUATION CALCULATOR - DEMO")
    print("="*80)
    
    # Initialize calculator
    calc = ValuationCalculator()
    
    # Example 1: Revenue-generating tech startup
    print("\n" + "="*80)
    print("EXAMPLE 1: Tech Startup with Revenue")
    print("="*80)
    
    result1 = calc.calculate_recommended_valuation(
        industry='Technology/Software',
        yearly_revenue=450,  # ₹450L
        gross_margin=75,
        net_margin=30,
        has_patent=True,
        team_quality=8
    )
    
    # Example 2: Assess a valuation ask
    print("\n" + "="*80)
    print("EXAMPLE 2: Assess Valuation Ask")
    print("="*80)
    
    assessment = calc.assess_ask(
        industry='Food and Beverage',
        yearly_revenue=200,  # ₹200L = ₹2Cr
        asked_valuation=2500  # ₹2500L = ₹25Cr
    )
    
    # Example 3: Pre-revenue startup
    print("\n" + "="*80)
    print("EXAMPLE 3: Pre-Revenue Startup")
    print("="*80)
    
    pre_revenue_val = calc.calculate_pre_revenue_valuation(
        industry='Medical/Health',
        team_pedigree=9,  # IIT/IIM founders with healthcare experience
        market_size=5000,  # ₹50Cr TAM
        product_stage='beta',
        traction=2000,  # 2000 users in beta
        has_patent=True
    )
    
    # Example 4: Fashion startup assessment
    print("\n" + "="*80)
    print("EXAMPLE 4: Fashion Startup Assessment")
    print("="*80)
    
    assessment2 = calc.assess_ask(
        industry='Beauty/Fashion',
        yearly_revenue=380,
        asked_valuation=250
    )
    
    print("\n" + "="*80)
    print("✅ VALUATION CALCULATOR DEMO COMPLETE")
    print("="*80)
    print("\nThis tool can be used to:")
    print("   • Calculate recommended valuations for revenue-generating startups")
    print("   • Assess if asked valuations are reasonable")
    print("   • Estimate pre-revenue startup valuations")
    print("   • Understand industry-specific benchmarks")
    print("\nBased on real Shark Tank India data and industry benchmarks.")
    print("="*80)
