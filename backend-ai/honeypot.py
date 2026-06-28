"""
Honeypot detection module to identify fake candidate profiles.
"""

def is_honeypot(candidate_dict):
    """
    Evaluates candidate data against logical honeypot filters to catch fake profiles.
    Returns True if the candidate is deemed a honeypot, False otherwise.
    
    Rules:
    1. Total experience > 50 years.
    2. A skill marked "Expert" with 0 duration.
    """
    try:
        # ---------------------------------------------------------
        # Rule 1: Extract candidate's total years of experience
        # ---------------------------------------------------------
        total_exp = 0.0
        exp_data = candidate_dict.get('career_history', [])
        
        if isinstance(exp_data, list):
            for exp in exp_data:
                if isinstance(exp, dict):
                    months = exp.get('duration_months', 0)
                    try:
                        total_exp += float(months) / 12.0
                    except (ValueError, TypeError):
                        pass
                        
        if total_exp > 50:
            return True
            
        # ---------------------------------------------------------
        # Rule 2: Skill marked "Expert" but 0 years of experience
        # ---------------------------------------------------------
        skills_data = candidate_dict.get('skills', [])
        if isinstance(skills_data, list):
            for skill in skills_data:
                if isinstance(skill, dict):
                    proficiency = str(skill.get('proficiency', '')).strip().lower()
                    months = skill.get('duration_months', None)
                    
                    if proficiency == 'expert':
                        try:
                            if months is not None and float(months) == 0.0:
                                return True
                        except (ValueError, TypeError):
                            pass
                            
        return False
        
    except Exception:
        # Fail gracefully if unexpected schema
        return False
