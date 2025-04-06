import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class ConditionScores:
    def __init__(self):
        # Maximum scores for each condition
        self.Dryness_Max_Score = 13
        self.Damage_Max_Score = 15
        self.Sensitivity_Max_Score = 4.4
        self.Sebum_Max_Score_oily = 4.2
        self.Sebum_Max_Score_dry = 4.2
        self.Flakes_Max_Score = 4.1

    def calculate_condition_scores(self, user_preferences: dict) -> dict:
        """Calculate condition scores from user preferences"""
        
        # Initialize scores dictionary
        scores = {
            'IssuesSC': 0,
            'hair_overalSC': 0,
            'daily_waterSC': 0,
            'HairFeelsSC': 0,
            'PorositySC': 0,
            'TreatmentSC': 0,
            'EnhancersSC': 0,
            'StyleSC': 0,
            'split_endsSC': 0,
            'heat_freqSC': 0,
            'scalp_feelingSC': 0,
            'scalp_flakySC': 0,
            'scalp_oilySC': 0,
            'scalp_drySC': 0
        }

        # Calculate PorositySC
        if "Easily absorbs water" in user_preferences['hair_behaviour']:
            scores['PorositySC'] = 2
        elif "Its bouncy and elastic" in user_preferences['hair_behaviour']:
            scores['PorositySC'] = 1
        elif "Repels moisture when wet" in user_preferences['hair_behaviour']:
            scores['PorositySC'] = 0

        # Calculate IssuesSC
        if any(issue in user_preferences['issue'].lower() for issue in ["scalp dryness", "breakage", "split", "thinning"]):
            scores['IssuesSC'] = 2

        # Calculate scalp_feelingSC
        if "Sensitive" in user_preferences['scalp_feeling'] or "Itchy" in user_preferences['scalp_feeling']:
            scores['scalp_feelingSC'] = 2
        elif "A little bit tight" in user_preferences['scalp_feeling']:
            scores['scalp_feelingSC'] = 1

        # Calculate scalp_flakySC
        flaky_map = {"Always": 3, "Usually": 2, "Rarely": 1, "No Not Me": 0}
        scores['scalp_flakySC'] = flaky_map.get(user_preferences['scalp_flaky'], 0)

        # Calculate scalp_oilySC
        if "My scalp gets dry" in user_preferences['oily_scalp']:
            scores['scalp_oilySC'] = 0
        elif "5 + Days" in user_preferences['oily_scalp']:
            scores['scalp_oilySC'] = 1
        elif "3 - 4 Days" in user_preferences['oily_scalp']:
            scores['scalp_oilySC'] = 2
        elif "1 - 2 Days" in user_preferences['oily_scalp']:
            scores['scalp_oilySC'] = 3

        # Calculate scalp_drySC
        if "Within hours" in user_preferences['dry_scalp']:
            scores['scalp_drySC'] = 3
        elif "1 - 2 Days" in user_preferences['dry_scalp']:
            scores['scalp_drySC'] = 2
        elif "3 - 5 Days" in user_preferences['dry_scalp']:
            scores['scalp_drySC'] = 1
        elif "My scalp gets oily" in user_preferences['dry_scalp']:
            scores['scalp_drySC'] = 0

        # Calculate TreatmentSC
        if "No not me" not in user_preferences['treatments']:
            scores['TreatmentSC'] = 1

        # Calculate totals
        total1 = (scores['IssuesSC'] + scores['hair_overalSC'] + 
                 scores['daily_waterSC'] + scores['HairFeelsSC'] + 
                 scores['scalp_flakySC'] + scores['split_endsSC'] + 
                 scores['scalp_drySC'])

        total2 = (scores['TreatmentSC'] + scores['EnhancersSC'] + 
                 scores['PorositySC'] + scores['StyleSC'] + 
                 scores['heat_freqSC'] + scores['split_endsSC'])

        total3 = scores['scalp_feelingSC']
        total4 = scores['scalp_oilySC']
        total5 = scores['scalp_drySC']
        total6 = scores['scalp_flakySC']

        # Calculate final percentages
        condition_scores = {
            'dryness': round((total1 * 100 / self.Dryness_Max_Score), 1),
            'damage': round((total2 * 100 / self.Damage_Max_Score), 1),
            'sensitivity': round((total3 * 100 / self.Sensitivity_Max_Score), 1),
            'sebum_oil': round((total4 * 100 / self.Sebum_Max_Score_oily), 1),
            'dry_scalp_score': round((total5 * 100 / self.Sebum_Max_Score_dry), 1),
            'flakes': round((total6 * 100 / self.Flakes_Max_Score), 1)
        }

        return condition_scores 