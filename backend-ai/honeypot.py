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
        # Rule 2: Senior title with < 3 years experience
        # ---------------------------------------------------------
        profile = candidate_dict.get('profile', {})
        current_title = str(profile.get('current_title', '')).lower()
        if any(kw in current_title for kw in ['senior', 'lead', 'principal']) and total_exp < 3.0:
            return True

        # ---------------------------------------------------------
        # Rule 3, 4, 5: Skills and Assessments
        # ---------------------------------------------------------
        skills_data = candidate_dict.get('skills', [])
        signals = candidate_dict.get('redrob_signals', {})
        assessments = signals.get('skill_assessment_scores', {}) if isinstance(signals, dict) else {}
        
        advanced_count = 0
        fake_duration_count = 0
        
        if isinstance(skills_data, list):
            for skill in skills_data:
                if isinstance(skill, dict):
                    proficiency = str(skill.get('proficiency', '')).strip().lower()
                    months = skill.get('duration_months', None)
                    name = str(skill.get('name', ''))
                    
                    if proficiency in ['advanced', 'expert']:
                        advanced_count += 1
                        
                        # Rule 3: Advanced/Expert with < 3 months duration (track count)
                        try:
                            if months is not None and float(months) < 3.0:
                                fake_duration_count += 1
                        except (ValueError, TypeError):
                            pass
                            
                        # Rule 4: Assessment score < 20 on advanced skill
                        if name in assessments:
                            score = assessments[name]
                            try:
                                if float(score) < 20.0:
                                    return True
                            except (ValueError, TypeError):
                                pass
                                
        if fake_duration_count > 1:
            return True
                                
        # Rule 5: > 12 advanced skills but total exp < 3 years
        if advanced_count > 12 and total_exp < 3.0:
            return True
            
        return False
        
    except Exception:
        # Fail gracefully if unexpected schema
        return False
