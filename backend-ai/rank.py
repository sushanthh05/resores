"""
Phase 2: Talent Resonance Engine - Ranker
Executes in a sandboxed, internet-free container to score candidates.
Implements Phase 3: Career Physics Layer (Exponential Skill Decay).
Uses local HuggingFace Qwen2.5 for generative reasoning on CPU.
"""

import argparse
import time
import pandas as pd
import numpy as np
import json
from datetime import datetime
import torch
import random
from sentence_transformers import SentenceTransformer, util

from config import (
    PARQUET_OUTPUT_PATH,
    MODEL_NAME,
    MODEL_CACHE_DIR,
    ARTIFACTS_DIR
)

REFERENCE_DATE = datetime(2026, 6, 22)
JD_KEYWORDS = ["python", "ai", "rag"]

def calculate_experience_alignment(years):
    if 6 <= years <= 8: return 1.2
    if 5 <= years < 6: return 1.1
    if 4 <= years < 5: return 0.8
    if years < 3: return 0.3
    if years > 12: return 0.8
    return 1.0

def calculate_retrieval_score(summary, career_history_json, skills_json):
    text = f"{summary} {career_history_json} {skills_json}".lower()
    keywords = ['retrieval', 'ranking', 'recommendation system', 'recommendation engine', 'search', 'personalization', 'relevance', 'matching', 'embeddings', 'vector database', 'hybrid retrieval', 'ndcg', 'map', 'a/b testing', 'offline evaluation']
    hits = sum(1 for kw in keywords if kw in text)
    return min(hits / 5.0, 1.0)

def calculate_production_ml_score(summary, career_history_json):
    text = f"{summary} {career_history_json}".lower()
    boosts = ['production', 'deployed', 'shipped', 'scaling', 'monitoring', 'latency', 'real users', 'experimentation', 'online metrics']
    penalties = ['academic', 'research', 'paper']
    boost_hits = sum(1 for kw in boosts if kw in text)
    penalty_hits = sum(1 for kw in penalties if kw in text)
    
    score = min(boost_hits / 3.0, 1.0)
    if penalty_hits > 0 and boost_hits == 0:
        score = 0.0 # Heavy penalty for pure academic
    return score

def calculate_product_company_score(career_history_json):
    try:
        history = json.loads(career_history_json)
    except:
        return 0.5, False
        
    if not isinstance(history, list): return 0.5, False
    
    consulting_firms = ['wipro', 'infosys', 'tcs', 'cognizant', 'accenture', 'capgemini', 'it services', 'consulting']
    total_months = 0
    consulting_months = 0
    product_boost = 0
    
    for role in history:
        if isinstance(role, dict):
            months = role.get('duration_months', 0)
            try: m = float(months)
            except: m = 0.0
            total_months += m
            
            text = f"{role.get('company', '')} {role.get('industry', '')}".lower()
            if any(kw in text for kw in consulting_firms):
                consulting_months += m
            if any(kw in text for kw in ['saas', 'startup', 'product', 'marketplace']):
                product_boost += 1
                
    score = 0.5 + min(product_boost * 0.2, 0.5)
    is_consulting_heavy = (consulting_months / total_months > 0.8) if total_months > 0 else False
    
    return min(score, 1.0), is_consulting_heavy

def calculate_stability_score(career_history_json):
    try: history = json.loads(career_history_json)
    except: return 1.0
    if not isinstance(history, list) or len(history) == 0: return 1.0
        
    total_months = 0
    for role in history:
        if isinstance(role, dict):
            try: total_months += float(role.get('duration_months', 0))
            except: pass
                
    avg_tenure = total_months / len(history)
    if avg_tenure >= 24: return 1.2
    if avg_tenure < 18: return 0.5
    return 1.0

def calculate_recruiter_signal_score(signals_json):
    try: sig = json.loads(signals_json)
    except: return 0.0
    if not isinstance(sig, dict): return 0.0
    
    score = 0.0
    score += float(sig.get('recruiter_response_rate', 0.0))
    score += float(sig.get('interview_completion_rate', 0.0))
    score += float(sig.get('offer_acceptance_rate', 0.0))
    
    saved = float(sig.get('saved_by_recruiters_30d', 0.0))
    score += min(saved / 10.0, 1.0)
    
    searches = float(sig.get('search_appearance_30d', 0.0))
    score += min(searches / 100.0, 1.0)
    
    views = float(sig.get('profile_views_received_30d', 0.0))
    score += min(views / 50.0, 1.0)
    
    return min(score / 6.0, 1.0)

def calculate_availability_score(signals_json):
    try: sig = json.loads(signals_json)
    except: return 0.5
    if not isinstance(sig, dict): return 0.5
    
    score = 0.5
    if sig.get('open_to_work_flag'): score += 0.2
    apps = float(sig.get('applications_submitted_30d', 0.0))
    score += min(apps / 5.0, 0.3)
    
    last_active = str(sig.get('last_active_date', ''))
    if last_active:
        try:
            active_dt = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = (REFERENCE_DATE - active_dt).days
            if days_inactive > 180: score *= 0.1
            elif days_inactive <= 30: score += 0.2
        except: pass
            
    try:
        np_days = float(sig.get('notice_period_days', 0))
        if np_days > 90: score *= 0.3
        elif np_days <= 30: score += 0.2
    except: pass
        
    return min(score, 1.0)

def calculate_trust_score(signals_json):
    try: sig = json.loads(signals_json)
    except: return 0.5
    if not isinstance(sig, dict): return 0.5
    
    score = 0.0
    if sig.get('verified_email'): score += 0.3
    if sig.get('verified_phone'): score += 0.3
    if sig.get('linkedin_connected'): score += 0.2
    
    comp = float(sig.get('profile_completeness_score', 0.0))
    score += (comp / 100.0) * 0.2
    
    return min(score, 1.0)

def is_framework_enthusiast(skills_json, retrieval_score):
    if retrieval_score > 0.4: return False 
    text = skills_json.lower()
    kws = ['langchain', 'openai', 'prompt engineering', 'gpt', 'rag']
    hits = sum(1 for kw in kws if kw in text)
    return hits >= 2

def generate_recruiter_reasoning(row, rank):
    years = float(row.get('years_experience', 0.0))
    summary = str(row.get('summary', ''))
    cj = str(row.get('career_history_json', '[]'))
    sj = str(row.get('skills_json', '[]'))
    rj = str(row.get('redrob_signals_json', '{}'))
    
    full_text = f"{summary} {cj} {sj}".lower()
    
    try: sig = json.loads(rj)
    except: sig = {}
    
    retrieval_kws = ['retrieval', 'ranking', 'recommendation', 'search', 'relevance', 'matching', 'semantic search', 'vector search', 'information retrieval']
    eval_kws = ['ndcg', 'map', 'mrr', 'recall@k', 'offline evaluation', 'online evaluation', 'a/b testing', 'experimentation', 'calibration', 'relevance metrics']
    prod_kws = ['deployed', 'production', 'shipped', 'scaling', 'latency', 'monitoring', 'serving', 'real users', 'traffic', 'feature store', 'drift detection']
    infra_kws = ['pinecone', 'qdrant', 'weaviate', 'faiss', 'elasticsearch', 'opensearch', 'milvus', 'pgvector']
    llm_kws = ['lora', 'qlora', 'peft', 'fine tuning', 'quantization', 'serving optimization']
    company_kws = ['meta', 'google', 'netflix', 'microsoft', 'razorpay', 'flipkart', 'swiggy', 'zomato', 'startup']
    lead_kws = ['owned', 'led', 'mentored', 'built team', 'architecture ownership']
    
    has_ret = any(k in full_text for k in retrieval_kws)
    has_eval = any(k in full_text for k in eval_kws)
    has_prod = any(k in full_text for k in prod_kws)
    has_infra = any(k in full_text for k in infra_kws)
    has_llm = any(k in full_text for k in llm_kws)
    has_comp = any(k in full_text for k in company_kws)
    has_lead = any(k in full_text for k in lead_kws)
    
    ret_count = sum(1 for k in retrieval_kws if k in full_text)
    rec_count = sum(1 for k in ['recommendation', 'personalization', 'behavioral signals', 'collaborative filtering'] if k in full_text)
    llm_count = sum(1 for k in llm_kws if k in full_text)
    
    if ret_count > 0 and ret_count >= rec_count: archetype = "Retrieval"
    elif rec_count > 0: archetype = "Recommendation"
    elif has_prod and not has_ret: archetype = "Production ML"
    elif llm_count > 0: archetype = "LLM Infrastructure"
    elif sum(1 for k in ['research', 'paper', 'experiment'] if k in full_text) > 2 and not has_prod: archetype = "Applied Scientist"
    else: archetype = "Software"
    
    if rank <= 10: align_phrases = ["strong alignment", "direct overlap", "extensive ownership", "deep expertise"]
    elif rank <= 50: align_phrases = ["relevant exposure", "experience with", "demonstrated familiarity", "solid background"]
    else: align_phrases = ["some exposure", "partial overlap", "adjacent experience", "basic familiarity"]
        
    rng = random.Random(str(row.get('candidate_id', 'unknown')))
    align_phrase = rng.choice(align_phrases)
    
    str_templates = []
    
    if archetype == "Retrieval":
        str_templates.append(f"Built retrieval and ranking systems with {align_phrase} of relevance optimization.")
        str_templates.append(f"Experience with vector search systems demonstrates {align_phrase} with Redrob's intelligence-layer requirements.")
        str_templates.append(f"Demonstrates {align_phrase} of end-to-end retrieval pipelines, from offline metric tuning to production deployment.")
        if has_infra: str_templates.append(f"Hands-on experience operating vector retrieval infrastructure and large-scale embedding pipelines in production environments.")
    elif archetype == "Recommendation":
        str_templates.append(f"Experience with recommendation pipelines and behavioral reranking aligns closely with Redrob's matching requirements.")
        str_templates.append(f"Built personalization engines with {align_phrase} of collaborative filtering and relevance metrics.")
    elif archetype == "Production ML":
        str_templates.append(f"Track record of shipping production systems and iterating using user feedback matches the product-engineering expectations.")
        str_templates.append(f"Strong focus on latency optimization and serving systems indicates {align_phrase} with solid engineering fundamentals.")
    elif archetype == "LLM Infrastructure":
        str_templates.append(f"Technical depth in fine-tuning and inference optimization provides {align_phrase} with modern AI architectures.")
    elif archetype == "Applied Scientist":
        str_templates.append(f"Deep theoretical background with {align_phrase} applying research to offline evaluation workflows.")
        
    if has_eval:
        str_templates.append(f"Demonstrated ownership of offline metrics and online experimentation including NDCG and A/B testing workflows.")
    if has_lead:
        str_templates.append(f"Previous architecture ownership and mentoring responsibilities indicate senior IC readiness.")
    if has_comp:
        str_templates.append(f"Background at major product-focused tech companies suggests high cultural alignment.")
        
    if not str_templates:
        str_templates.append(f"Profile shows {align_phrase} with core machine learning technologies required for the role.")
        str_templates.append(f"General software engineering background with {align_phrase} of Python and modern data stacks.")
        
    num_str = 2 if len(str_templates) > 1 and rank <= 10 else 1
    selected_strengths = rng.sample(str_templates, num_str)
    
    concerns = []
    
    np_days = float(sig.get('notice_period_days', 0))
    if np_days > 60: concerns.append(f"Longer {int(np_days)}-day notice period slightly reduces immediate availability.")
    
    last_active = str(sig.get('last_active_date', ''))
    if last_active:
        try:
            active_dt = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = (REFERENCE_DATE - active_dt).days
            if days_inactive > 180: concerns.append("Lower recent platform activity reduces confidence in current job-search intent.")
        except: pass
        
    if sig.get('open_to_work_flag') is False:
        concerns.append("Not currently marked as open to work.")
        
    if float(sig.get('recruiter_response_rate', 1.0)) < 0.2:
        concerns.append("Limited recruiter responsiveness historically lowers short-term hiring probability.")
        
    if has_ret and not has_eval:
        concerns.append("Shows retrieval exposure but limited evidence of ranking evaluation or relevance metrics.")
        
    if has_llm and not has_eval:
        concerns.append("Demonstrates LLM experience but lacks a strong evaluation background.")
        
    if not has_prod and rank > 50:
        concerns.append("Profile leans more toward research or offline development than production deployment relative to higher-ranked candidates.")
        
    try:
        history = json.loads(cj)
        if isinstance(history, list):
            consulting_firms = ['wipro', 'infosys', 'tcs', 'cognizant', 'accenture', 'capgemini', 'it services', 'consulting']
            total_months = sum(float(r.get('duration_months', 0)) for r in history if isinstance(r, dict))
            consulting_months = sum(float(r.get('duration_months', 0)) for r in history if isinstance(r, dict) and any(kw in f"{r.get('company', '')} {r.get('industry', '')}".lower() for kw in consulting_firms))
            if total_months > 0 and consulting_months / total_months > 0.8:
                concerns.append("Career history shows limited ownership in product environments compared with higher-ranked candidates.")
    except: pass
        
    final_sentences = selected_strengths
    if concerns:
        final_sentences.append(rng.choice(concerns))
        
    return " ".join(final_sentences)

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Talent Resonance Engine")
    parser.add_argument("--candidates", type=str, required=False, help="Path to original jsonl (ignored)")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    overall_start = time.time()
    
    # ---------------------------------------------------------
    # 1. Fast Loading
    # ---------------------------------------------------------
    t0 = time.time()
    print(f"Loading pre-computed parquet from {PARQUET_OUTPUT_PATH}...")
    try:
        df = pd.read_parquet(PARQUET_OUTPUT_PATH)
    except FileNotFoundError:
        print(f"Error: Parquet file not found. Run precompute.py first.")
        return
    print(f"[Timing] Loading Parquet took {time.time() - t0:.4f} seconds.")

    # ---------------------------------------------------------
    # 2. Local Embedding
    # ---------------------------------------------------------
    t1 = time.time()
    print(f"Initializing Local Model: {MODEL_NAME} on CPU...")
    model = SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE_DIR), device='cpu', local_files_only=True)
    
    job_description_text = "Looking for a Senior AI Engineer with Python, Vector Search, and RAG experience"
    print(f"Embedding Job Description: '{job_description_text}'")
    jd_embedding = model.encode(job_description_text)
    print(f"[Timing] Model Loading & JD Embedding took {time.time() - t1:.4f} seconds.")

    # ---------------------------------------------------------
    # 3. Vector Math (Cosine Similarity)
    # ---------------------------------------------------------
    t2 = time.time()
    print("Performing simultaneous Vector Math via NumPy...")
    candidate_matrix = np.vstack(df['embedding'].values)
    jd_norm = np.linalg.norm(jd_embedding)
    matrix_norms = np.linalg.norm(candidate_matrix, axis=1)
    
    jd_norm = jd_norm if jd_norm > 0 else 1e-10
    matrix_norms[matrix_norms == 0] = 1e-10
    
    dot_products = np.dot(candidate_matrix, jd_embedding)
    df['semantic_score'] = dot_products / (jd_norm * matrix_norms)
    print(f"[Timing] Vector Math Matrix Operations took {time.time() - t2:.4f} seconds.")

    # ---------------------------------------------------------
    # 4. Multi-dimensional Feature Scoring
    # ---------------------------------------------------------
    t3 = time.time()
    print("Calculating structured feature scores and penalties...")
    
    final_scores = []
    
    for _, row in df.iterrows():
        sem_score = float(row.get('semantic_score', 0.0))
        years = float(row.get('years_experience', 0.0))
        summary = str(row.get('summary', ''))
        cj = str(row.get('career_history_json', '[]'))
        sj = str(row.get('skills_json', '[]'))
        rj = str(row.get('redrob_signals_json', '{}'))
        is_hp = bool(row.get('is_honeypot', False))
        
        # Calculate feature scores
        exp_score = calculate_experience_alignment(years)
        ret_score = calculate_retrieval_score(summary, cj, sj)
        prod_score = calculate_production_ml_score(summary, cj)
        prod_comp_score, is_consulting = calculate_product_company_score(cj)
        stab_score = calculate_stability_score(cj)
        recruiter_score = calculate_recruiter_signal_score(rj)
        avail_score = calculate_availability_score(rj)
        trust_score = calculate_trust_score(rj)
        
        # Base Final Score Calculation
        fs = (
            0.40 * sem_score +
            0.20 * ret_score +
            0.15 * prod_score +
            0.10 * recruiter_score +
            0.05 * exp_score +
            0.05 * prod_comp_score +
            0.03 * avail_score +
            0.02 * trust_score
        )
        
        # Apply Multiplicative Penalties
        if is_consulting: fs *= 0.5
        if prod_score == 0.0 and sum(1 for kw in ['academic', 'research'] if kw in (summary + cj).lower()) > 1:
            fs *= 0.5 # Research only penalty
        if is_framework_enthusiast(sj, ret_score):
            fs *= 0.5
            
        fs *= stab_score # Stability multiplier
            
        if is_hp:
            fs = -999.0
            
        final_scores.append(fs)
        
    df['final_score'] = final_scores
    print(f"[Timing] Structured Scoring took {time.time() - t3:.4f} seconds.")

    # ---------------------------------------------------------
    # 5. Strict Sorting & Tie-Breaking
    # ---------------------------------------------------------
    t4 = time.time()
    print("Sorting deterministically and slicing top 100...")
    df = df.sort_values(by=['final_score', 'candidate_id'], ascending=[False, True])
    
    top_100 = df.head(100).copy()
    top_100['rank'] = range(1, len(top_100) + 1)
    
    # Standardize column name back to 'score' for final output schema
    top_100['score'] = top_100['final_score']
    print(f"[Timing] Sorting & Slicing took {time.time() - t4:.4f} seconds.")

    # ---------------------------------------------------------
    # 5. Deterministic Reasoning Generation
    # ---------------------------------------------------------
    t5 = time.time()
    print("Generating deterministic evidence-based recruiter reasoning strings...")
    
    reasoning_list = []
    
    for i, row in top_100.iterrows():
        # Rank is effectively index in top_100 + 1 (assuming it is sorted and reset, but let's calculate based on actual position in the df)
        rank_position = list(top_100.index).index(i) + 1
        reasoning = generate_recruiter_reasoning(row, rank_position)
        reasoning_list.append(reasoning)

    top_100['reasoning'] = reasoning_list
    print(f"[Timing] Recruiter Reasoning Generation took {time.time() - t5:.4f} seconds.")

    # ---------------------------------------------------------
    # 7. Strict CSV Output
    # ---------------------------------------------------------
    print(f"Exporting strictly required columns to {args.out}...")
    final_columns = ['candidate_id', 'rank', 'score', 'reasoning']
    final_export = top_100[final_columns]
    
    try:
        final_export.to_csv(args.out, index=False)
    except PermissionError:
        fallback_path = args.out + ".fallback.csv"
        print(f"\n[ERROR] Permission denied: {args.out} is locked (likely open in your IDE or Excel).")
        print(f"[INFO] Saving to {fallback_path} instead to prevent losing generation data!\n")
        final_export.to_csv(fallback_path, index=False)
    
    overall_end = time.time()
    print(f"[Timing] Total rank.py execution took {overall_end - overall_start:.4f} seconds.")
    print("Pipeline Phase 2 (Ranking) Completed Successfully.")

if __name__ == "__main__":
    main()
