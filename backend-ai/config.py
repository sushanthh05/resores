import os
import pathlib

# Resolve the directory containing this config file (i.e., backend-ai/)
BACKEND_AI_DIR = pathlib.Path(__file__).parent.resolve()

# Workspace root is the parent of backend-ai
WORKSPACE_ROOT = BACKEND_AI_DIR.parent

# Allow environment variable override for target dataset
DATASET_TARGET = os.getenv("DATASET_TARGET", "production")

# Datasets and Artifacts directories
DATASET_DIR = WORKSPACE_ROOT / "dataset"
ARTIFACTS_DIR = BACKEND_AI_DIR / "artifacts"
MODEL_CACHE_DIR = BACKEND_AI_DIR / "model_cache"

# File paths
if DATASET_TARGET == "production":
    SAMPLE_CANDIDATES_PATH = DATASET_DIR / "candidates.jsonl"
else:
    SAMPLE_CANDIDATES_PATH = DATASET_DIR / "sample_candidates.json"

PARQUET_OUTPUT_PATH = ARTIFACTS_DIR / "candidates_enriched.parquet"

# Model Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
