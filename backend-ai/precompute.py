"""
Phase 1: The Offline Pre-Computation Pipeline
Reads candidate data, applies honeypot filters, generates semantic embeddings locally,
and saves the output to a compressed Parquet file with accurate schema mapping.
Downloads LLM offline.
"""

import pandas as pd
import json
import logging
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyarrow

from config import (
    SAMPLE_CANDIDATES_PATH,
    PARQUET_OUTPUT_PATH,
    MODEL_NAME,
    MODEL_CACHE_DIR,
    ARTIFACTS_DIR
)
from honeypot import is_honeypot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Reference date for "Present" calculations (ensuring deterministic behavior)
REFERENCE_DATE = datetime(2026, 6, 22)

def calculate_years_experience(career_history):
    """Calculates total experience natively using exact dates between start and end."""
    total_days = 0
    if not isinstance(career_history, list):
        return 0.0
        
    for role in career_history:
        if not isinstance(role, dict):
            continue
        start = role.get('start_date')
        end = role.get('end_date')
        if not start:
            continue
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            if end:
                end_dt = datetime.strptime(end, "%Y-%m-%d")
            else:
                end_dt = REFERENCE_DATE
            
            days = (end_dt - start_dt).days
            if days > 0:
                total_days += days
        except Exception:
            pass
            
    # Return as a clean float
    return round(total_days / 365.25, 1)

def build_semantic_text(row_dict):
    """
    Combines job_title, skills, and experience into a single rich text string.
    """
    profile = row_dict.get('profile', {})
    
    # Extract job title accurately from schema
    job_title = str(profile.get('current_title', '')).strip()
    if job_title.lower() == 'nan':
        job_title = ""
    
    # Extract skills
    skills = row_dict.get('skills', [])
    skills_str = ""
    if isinstance(skills, list):
        parsed_skills = []
        for s in skills:
            if isinstance(s, dict):
                parsed_skills.append(str(s.get('name', '')))
        skills_str = ", ".join([s for s in parsed_skills if s])
        
    # Extract experience via career_history
    career_history = row_dict.get('career_history', [])
    exp_summary = ""
    if isinstance(career_history, list):
        exp_strs = []
        for exp in career_history:
            if isinstance(exp, dict):
                title = exp.get('title', '')
                company = exp.get('company', '')
                if title and company:
                    exp_strs.append(f"{title} at {company}")
                elif title:
                    exp_strs.append(title)
                elif company:
                    exp_strs.append(company)
        exp_summary = " | ".join([e for e in exp_strs if e])
        
    parts = []
    if job_title:
        parts.append(f"Title: {job_title}")
    if skills_str:
        parts.append(f"Skills: {skills_str}")
    if exp_summary:
        parts.append(f"Experience: {exp_summary}")
        
    return ". ".join(parts) + "." if parts else ""

def main():
    logging.info("Starting Offline Pre-Computation Pipeline...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading candidate data from {SAMPLE_CANDIDATES_PATH}")
    try:
        if str(SAMPLE_CANDIDATES_PATH).endswith('.jsonl'):
            # Stream JSONL line-by-line
            candidates = []
            with open(SAMPLE_CANDIDATES_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        candidates.append(json.loads(line))
        else:
            with open(SAMPLE_CANDIDATES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'candidates' in data:
                    candidates = data['candidates']
                elif isinstance(data, list):
                    candidates = data
                else:
                    logging.error("Unsupported JSON schema.")
                    return
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return
        
    df = pd.DataFrame(candidates)
    if df.empty: return
    
    logging.info("Applying honeypot filters, extracting exact metadata, and generating semantic text...")
    
    semantic_texts = []
    is_hp_flags = []
    current_titles = []
    years_experiences = []
    skills_lists = []
    career_history_jsons = []
    
    # We iterate using tqdm to process complex nested objects safely
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        row_dict = row.to_dict()
        
        # 1. Honeypot Check
        is_hp_flags.append(is_honeypot(row_dict))
        
        # 2. Semantic Text Generation
        semantic_texts.append(build_semantic_text(row_dict))
        
        # 3. Profile Metadata Extraction
        profile = row_dict.get('profile', {})
        current_titles.append(profile.get('current_title', 'Unknown'))
        
        # 4. Career History & Experience Calculation
        career_history = row_dict.get('career_history', [])
        years_experiences.append(calculate_years_experience(career_history))
        # Serialize the array so we can safely persist it in the Parquet file for rank.py to parse later
        career_history_jsons.append(json.dumps(career_history))
        
        # 5. Cleaned Skills List
        skills = row_dict.get('skills', [])
        extracted_skills = []
        if isinstance(skills, list):
            extracted_skills = [s.get('name') for s in skills if isinstance(s, dict) and s.get('name')]
        skills_lists.append(extracted_skills)
        
    # Append all processed columns to DataFrame
    df['semantic_text'] = semantic_texts
    df['is_honeypot'] = is_hp_flags
    df['current_title'] = current_titles
    df['years_experience'] = years_experiences
    df['skills_list'] = skills_lists
    df['career_history_json'] = career_history_jsons
    
    logging.info(f"Honeypot flagged {sum(is_hp_flags)} candidates.")
    
    logging.info(f"Initializing SentenceTransformer: {MODEL_NAME} on CPU")
    model = SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE_DIR), device='cpu')
        
    logging.info("Embedding semantic text into dense vectors...")
    embeddings = model.encode(df['semantic_text'].tolist(), show_progress_bar=True)
    df['embedding'] = embeddings.tolist()
    
    if 'candidate_id' not in df.columns:
        df['candidate_id'] = df.get('id', df.index)
            
    # Include all our new metadata columns
    final_cols = [
        'candidate_id', 'semantic_text', 'is_honeypot', 'embedding',
        'current_title', 'years_experience', 'skills_list', 'career_history_json'
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    final_df = df[final_cols]
    
    logging.info(f"Saving artifacts to {PARQUET_OUTPUT_PATH}")
    final_df.to_parquet(PARQUET_OUTPUT_PATH, engine='pyarrow', compression='snappy')
    logging.info("Embeddings and metadata successfully saved.")
    
    # ---------------------------------------------------------
    # Generative AI Model Download for Phase 2
    # ---------------------------------------------------------
    logging.info("Downloading Generative LLM (Qwen2.5-0.5B-Instruct) for Offline Reasoning...")
    llm_cache_path = ARTIFACTS_DIR / "llm_cache"
    llm_cache_path.mkdir(parents=True, exist_ok=True)
    
    try:
        qwen_name = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(qwen_name)
        tokenizer.save_pretrained(llm_cache_path)
        
        causal_model = AutoModelForCausalLM.from_pretrained(qwen_name)
        causal_model.save_pretrained(llm_cache_path)
        
        logging.info(f"Successfully downloaded and saved generative LLM to {llm_cache_path}")
    except Exception as e:
        logging.error(f"Failed to download generative LLM: {e}")

    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
