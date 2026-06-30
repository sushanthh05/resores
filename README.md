# Talent Resonance Engine - Candidate Ranking Pipeline

Welcome to the Redrob AI Hackathon candidate ranking project. This pipeline is designed to rank software engineering candidates against a specific Job Description (JD) using a highly constrained, CPU-only environment.

We abandoned generative LLM approaches (like Qwen) due to strict constraints:
- **No external APIs** (offline sandbox)
- **CPU only**
- **<16GB RAM limit**
- **<5-minute execution time limit**
- **Zero hallucinations allowed**

Instead, we built a highly optimized **Extractive RAG (Retrieval-Augmented Generation) Architecture** for evidence-based candidate reasoning.

## What We Built

The architecture is split into two strict phases:

### 1. Phase 1: Offline Precomputation (`backend-ai/precompute.py`)
This script processes the raw JSONL candidate data. It performs heavy lifting such as:
- Parsing candidate JSONs deterministically.
- Detecting and filtering out "honeypot" candidates (consultants, pure researchers, etc.).
- Generating dense vector embeddings for candidate profiles using the `all-MiniLM-L6-v2` model.
- Saving everything to an optimized `candidates_enriched.parquet` file for ultra-fast loading later.

### 2. Phase 2: Live Ranking & Reasoning (`backend-ai/rank.py`)
This is the core script that runs within the 5-minute sandbox limit:
- **Vector Math**: Loads the precomputed parquet and uses vectorized NumPy operations to calculate cosine similarity between candidate embeddings and the JD embedding.
- **Structured Scoring**: Applies multi-dimensional scoring (semantic match, years of experience, product vs. research background, stability, recruiter signals).
- **Extractive Reasoning**: Rather than generating a summary, it extracts the exact sentences from the candidate's profile that support their high rank. It uses pattern matching and multipliers to banish boilerplate statements (e.g., *"Looking for my next move"*) and aggressively rewards measurable achievements (e.g., *"Built a ranking pipeline serving 50M queries/month"*).

## Setup & Installation

Before running the pipeline, you must install the required dependencies. The project uses standard data science and machine learning libraries (Pandas, NumPy, PyTorch, Sentence-Transformers).

1. **Navigate to the backend directory:**
   ```powershell
   cd backend-ai
   ```
2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install the required packages:**
   ```powershell
   pip install -r requirements.txt
   ```

*Note: The first time you run the pipeline, the `all-MiniLM-L6-v2` model will be downloaded locally. All subsequent runs will load it instantly from the local cache.*

## How to Run the Pipeline

Ensure your virtual environment is active in the `backend-ai` directory:
```powershell
cd backend-ai
.\venv\Scripts\activate
```

**Step 1: Run Precomputation (Offline/One-time)**
```powershell
python precompute.py --dataset ../dataset/candidates.jsonl --out artifacts/candidates_enriched.parquet
```

**Step 2: Run Live Ranking**
```powershell
python rank.py --candidates ../dataset/candidates.jsonl --out ../team_akshay.csv
```

## Expected Outputs

The primary output is the `team_akshay.csv` file generated in the root directory. 

The CSV will strictly contain the following headers:
- `candidate_id`: The unique identifier.
- `rank`: The candidate's final rank (1 to 100).
- `score`: The final computed alignment score.
- `reasoning`: Highly dense, factual, extracted sentences explaining exactly *why* the candidate was selected (highlighting measurable achievements, product ownership, and critical recruiter signals).

*Note: For the reasoning column, we enforce a strict max-80 word limit, ensuring high information density for human recruiters.*
