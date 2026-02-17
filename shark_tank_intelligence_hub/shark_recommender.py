"""
SHARK RECOMMENDATION ENGINE
Recommend optimal shark combinations for startups based on industry, stage, and founder profile
"""

import pandas as pd
import numpy as np
import json
from itertools import combinations

class SharkRecommender:
    """
    Recommend best shark combinations for a startup based on:
    - Industry fit
    - Historical success rates
    - Co-investment patterns
    - Founder demographics
    """
    
    def __init__(self, 
                 data_path='data/processed/processed_data_full.csv',
                 profiles_path='reports/shark_profiles.json',
                 co_matrix_path='reports/co_investment_matrix.csv',
                 affinity_path='reports/shark_industry_affinity.csv'):
        """Load data and initialize recommender"""
        
        print("🦈 Initializing Shark Recommender...")
        
        # Load data
        self.df = pd.read_csv(data_path)
        
        # Load shark profiles
        with open(profiles_path, 'r') as f:
            self.shark_profiles = json.load(f)
        
        # Load co-investment matrix
        self.co_matrix = pd.read_csv(co_matrix_path, index_col=0)
        
        # Load affinity matrix
        self.affinity = pd.read_csv(affinity_path, index_col=0)
        
        self.sharks = list(self.shark_profiles.keys())
        
        print(f"✅ Loaded data for {len(self.sharks)} sharks")
        print(f"   Total startups: {len(self.df)}")
        print(f"   Industries: {len(self.affinity.index)}")
    
    def calculate_shark_score(self, shark, startup_profile):
        """
        Calculate individual shark score for a startup
        
        Parameters:
        -----------
        shark : str
            Shark name
        startup_profile : dict
            Startup characteristics
        
        Returns:
        --------
        float : Score 0-1
        """
        industry = startup_profile.get('industry', 'Technology/Software')
        founder_gender = startup_profile.get('founder_gender', 'male')
        revenue = startup_profile.get('revenue', 0)
        stage = startup_profile.get('stage', 'early')
        
        score = 0.0
        
        # 1. Industry Affinity (40% weight)
        if industry in self.affinity.index and shark in self.affinity.columns:
            industry_affinity = self.affinity.loc[industry, shark] / 100
            score += 0.4 * industry_affinity
        else:
            score += 0.4 * 0.1  # Default low affinity
        
        # 2. Historical Success Rate (30% weight)
        inv_col = f'{shark} Investment Amount'
        if inv_col in self.df.columns:
            shark_deals = self.df[self.df[inv_col] > 0]
            industry_deals = shark_deals[shark_deals['Industry'] == industry]
            
            if len(industry_deals) > 0:
                success_rate = industry_deals['Accepted Offer'].mean()
            else:
                # Use overall success rate
                success_rate = shark_deals['Accepted Offer'].mean() if len(shark_deals) > 0 else 0.5
            
            score += 0.3 * success_rate
        else:
            score += 0.3 * 0.5
        
        # 3. Gender Alignment Bonus (15% weight)
        gender_score = 0.5  # Default
        if founder_gender == 'female':
            if shark in ['Namita', 'Vineeta']:
                gender_score = 0.8  # Female sharks may have affinity for female founders
        score += 0.15 * gender_score
        
        # 4. Stage Preference (15% weight)
        # Estimate based on average revenue of shark's portfolio
        avg_revenue = self.shark_profiles[shark].get('avg_startup_revenue', 500)
        
        if stage == 'pre-revenue':
            stage_score = 0.7 if avg_revenue < 300 else 0.4
        elif stage == 'early':
            stage_score = 0.8 if avg_revenue < 500 else 0.6
        elif stage == 'growth':
            stage_score = 0.9 if avg_revenue > 300 else 0.5
        else:
            stage_score = 0.5
        
        score += 0.15 * stage_score
        
        return min(score, 1.0)  # Cap at 1.0
    
    def calculate_synergy(self, shark_list):
        """
        Calculate synergy score for a combination of sharks
        
        Parameters:
        -----------
        shark_list : list
            List of shark names
        
        Returns:
        --------
        float : Synergy score 0-1
        """
        if len(shark_list) == 1:
            return 0.0
        
        total_synergy = 0.0
        pair_count = 0
        
        # Calculate pairwise synergy
        for i in range(len(shark_list)):
            for j in range(i+1, len(shark_list)):
                s1, s2 = shark_list[i], shark_list[j]
                
                # Co-investment count
                if s1 in self.co_matrix.index and s2 in self.co_matrix.columns:
                    co_investments = self.co_matrix.loc[s1, s2]
                    # Normalize by max co-investments (assume ~50 is high)
                    synergy = min(co_investments / 50, 1.0)
                    total_synergy += synergy
                    pair_count += 1
        
        return total_synergy / pair_count if pair_count > 0 else 0.0
    
    def recommend_sharks(self, startup_profile, top_n=10, max_sharks=3):
        """
        Recommend best shark combinations for a startup
        
        Parameters:
        -----------
        startup_profile : dict
            Startup characteristics:
            - industry: str (required)
            - founder_gender: str ('male', 'female', 'mixed')
            - revenue: float (in lakhs)
            - stage: str ('pre-revenue', 'early', 'growth', 'mature')
            - valuation: float (in lakhs)
        top_n : int
            Number of recommendations to return
        max_sharks : int
            Maximum sharks per combination (1-3)
        
        Returns:
        --------
        list : Top N recommendations with scores
        """
        
        # Calculate individual shark scores
        shark_scores = {}
        for shark in self.sharks:
            shark_scores[shark] = self.calculate_shark_score(shark, startup_profile)
        
        # Generate all combinations
        all_combinations = []
        
        # Single sharks
        for shark in self.sharks:
            all_combinations.append({
                'sharks': [shark],
                'individual_scores': {shark: shark_scores[shark]},
                'base_score': shark_scores[shark],
                'synergy': 0.0,
                'total_score': shark_scores[shark]
            })
        
        # Pairs
        if max_sharks >= 2:
            for s1, s2 in combinations(self.sharks, 2):
                base_score = (shark_scores[s1] + shark_scores[s2]) / 2
                synergy = self.calculate_synergy([s1, s2])
                total_score = base_score + (0.2 * synergy)  # 20% weight to synergy
                
                all_combinations.append({
                    'sharks': [s1, s2],
                    'individual_scores': {s1: shark_scores[s1], s2: shark_scores[s2]},
                    'base_score': base_score,
                    'synergy': synergy,
                    'total_score': total_score
                })
        
        # Trios
        if max_sharks >= 3:
            for s1, s2, s3 in combinations(self.sharks, 3):
                base_score = (shark_scores[s1] + shark_scores[s2] + shark_scores[s3]) / 3
                synergy = self.calculate_synergy([s1, s2, s3])
                total_score = base_score + (0.15 * synergy)  # 15% weight to synergy
                
                all_combinations.append({
                    'sharks': [s1, s2, s3],
                    'individual_scores': {s1: shark_scores[s1], s2: shark_scores[s2], s3: shark_scores[s3]},
                    'base_score': base_score,
                    'synergy': synergy,
                    'total_score': total_score
                })
        
        # Sort by total score
        all_combinations.sort(key=lambda x: x['total_score'], reverse=True)
        
        return all_combinations[:top_n]
    
    def explain_recommendation(self, recommendation, startup_profile):
        """
        Provide detailed explanation for a recommendation
        
        Parameters:
        -----------
        recommendation : dict
            Recommendation from recommend_sharks()
        startup_profile : dict
            Startup profile
        
        Returns:
        --------
        str : Explanation text
        """
        sharks = recommendation['sharks']
        sharks_str = " + ".join(sharks)
        
        explanation = f"\n{'='*70}\n"
        explanation += f"🦈 RECOMMENDATION: {sharks_str}\n"
        explanation += f"{'='*70}\n"
        explanation += f"Total Score: {recommendation['total_score']:.3f}\n"
        explanation += f"Base Score: {recommendation['base_score']:.3f}\n"
        explanation += f"Synergy Bonus: {recommendation['synergy']:.3f}\n\n"
        
        explanation += "📊 Individual Shark Scores:\n"
        for shark in sharks:
            score = recommendation['individual_scores'][shark]
            explanation += f"   {shark:10s}: {score:.3f}\n"
        
        explanation += "\n💡 Why This Combination:\n"
        
        # Industry fit
        industry = startup_profile.get('industry', 'Unknown')
        explanation += f"   • Industry: {industry}\n"
        for shark in sharks:
            if industry in self.affinity.index and shark in self.affinity.columns:
                affinity = self.affinity.loc[industry, shark]
                explanation += f"     - {shark}: {affinity:.1f}% of portfolio in {industry}\n"
        
        # Partnership strength
        if len(sharks) > 1:
            explanation += f"\n   • Partnership History:\n"
            for i in range(len(sharks)):
                for j in range(i+1, len(sharks)):
                    s1, s2 = sharks[i], sharks[j]
                    if s1 in self.co_matrix.index and s2 in self.co_matrix.columns:
                        co_inv = int(self.co_matrix.loc[s1, s2])
                        explanation += f"     - {s1} ↔ {s2}: {co_inv} co-investments\n"
        
        # Shark specializations
        explanation += f"\n   • Shark Expertise:\n"
        for shark in sharks:
            profile = self.shark_profiles[shark]
            top_ind = list(profile['top_industries'].keys())[0] if profile['top_industries'] else 'Various'
            explanation += f"     - {shark}: {profile['total_deals']} deals, specializes in {top_ind}\n"
        
        explanation += f"\n{'='*70}\n"
        
        return explanation


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("🦈 SHARK RECOMMENDATION ENGINE - DEMO")
    print("="*80)
    
    # Initialize recommender
    recommender = SharkRecommender()
    
    # Example 1: HealthTech startup, female founder
    print("\n" + "="*80)
    print("EXAMPLE 1: HealthTech Startup (Female Founder)")
    print("="*80)
    
    startup1 = {
        'industry': 'Medical/Health',
        'founder_gender': 'female',
        'revenue': 200,
        'stage': 'early',
        'valuation': 2000
    }
    
    recommendations1 = recommender.recommend_sharks(startup1, top_n=5)
    
    for i, rec in enumerate(recommendations1, 1):
        sharks_str = " + ".join(rec['sharks'])
        print(f"\n{i}. {sharks_str}")
        print(f"   Total Score: {rec['total_score']:.3f} | Base: {rec['base_score']:.3f} | Synergy: {rec['synergy']:.3f}")
    
    # Detailed explanation for top recommendation
    print(recommender.explain_recommendation(recommendations1[0], startup1))
    
    # Example 2: Food & Beverage startup, male founder
    print("\n" + "="*80)
    print("EXAMPLE 2: Food & Beverage Startup (Male Founder)")
    print("="*80)
    
    startup2 = {
        'industry': 'Food and Beverage',
        'founder_gender': 'male',
        'revenue': 500,
        'stage': 'growth',
        'valuation': 3500
    }
    
    recommendations2 = recommender.recommend_sharks(startup2, top_n=5)
    
    for i, rec in enumerate(recommendations2, 1):
        sharks_str = " + ".join(rec['sharks'])
        print(f"\n{i}. {sharks_str}")
        print(f"   Total Score: {rec['total_score']:.3f} | Base: {rec['base_score']:.3f} | Synergy: {rec['synergy']:.3f}")
    
    print(recommender.explain_recommendation(recommendations2[0], startup2))
    
    # Example 3: Tech startup, pre-revenue
    print("\n" + "="*80)
    print("EXAMPLE 3: Technology Startup (Pre-Revenue)")
    print("="*80)
    
    startup3 = {
        'industry': 'Technology/Software',
        'founder_gender': 'mixed',
        'revenue': 0,
        'stage': 'pre-revenue',
        'valuation': 1500
    }
    
    recommendations3 = recommender.recommend_sharks(startup3, top_n=5)
    
    for i, rec in enumerate(recommendations3, 1):
        sharks_str = " + ".join(rec['sharks'])
        print(f"\n{i}. {sharks_str}")
        print(f"   Total Score: {rec['total_score']:.3f} | Base: {rec['base_score']:.3f} | Synergy: {rec['synergy']:.3f}")
    
    print(recommender.explain_recommendation(recommendations3[0], startup3))
    
    print("\n" + "="*80)
    print("✅ SHARK RECOMMENDER DEMO COMPLETE")
    print("="*80)
    print("\nThis tool can be used to:")
    print("   • Recommend optimal shark combinations for any startup")
    print("   • Consider industry fit, success rates, and partnerships")
    print("   • Account for founder demographics and startup stage")
    print("   • Provide detailed explanations for recommendations")
    print("\nBased on real Shark Tank India co-investment patterns and success data.")
    print("="*80)
