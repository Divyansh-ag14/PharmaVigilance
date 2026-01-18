"""
============================================
Pharma Vigilance - Configuration
============================================
Central configuration file for all settings.
Store your API keys in a .env file (never commit to git!)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# PROJECT PATHS
# ============================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / "cache"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ============================================
# DATA FILES
# ============================================
TRAIN_DATA_PATH = DATA_DIR / "drugLibTrain_raw.tsv"
TEST_DATA_PATH = DATA_DIR / "drugLibTest_raw.tsv"

# ============================================
# UF API CONFIGURATION (University of Florida)
# ============================================
# University API endpoint
UF_API_BASE = "https://api.ai.it.ufl.edu"

# API Key - Get from environment or set directly
UF_API_KEY = os.getenv("UF_API_KEY", "sk-k1-LJhxz8tVEZogMT88alQ")

# Available models at UF:
# - llama-3.1-70b-instruct (recommended - best quality)
# - llama-3.3-70b-instruct (newer version)
# - llama-3.1-8b-instruct (faster, less accurate)
# - mistral-7b-instruct
# - mistral-small-3.1
# - codestral-22b
# - gemma-3-27b-it
# - gpt-oss-20b / gpt-oss-120b
# - granite-3.3-8b-instruct
LLM_MODEL = "llama-3.1-70b-instruct"

# Fallback models if primary fails
FALLBACK_MODELS = [
    "llama-3.3-70b-instruct",
    "llama-3.1-8b-instruct",
    "mistral-small-3.1"
]

# ============================================
# LEGACY API KEYS (not used with UF API)
# ============================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============================================
# PROCESSING SETTINGS
# ============================================
# Batch size for LLM processing (to manage rate limits)
BATCH_SIZE = 10

# Maximum reviews to process (for testing, set to None for all)
MAX_REVIEWS = None

# Rate limiting (requests per minute)
RATE_LIMIT_RPM = 30

# Delay between API calls (seconds)
API_DELAY = 0.5

# ============================================
# ADVERSE EVENT CATEGORIES
# ============================================
# Standard severity levels for classification
SEVERITY_LEVELS = [
    "Mild",           # Minor discomfort, no intervention needed
    "Moderate",       # Noticeable impact, may need intervention
    "Severe",         # Significant impact, intervention required
    "Life-threatening" # Immediate medical attention required
]

# Common adverse event categories
ADVERSE_EVENT_CATEGORIES = [
    "Gastrointestinal",
    "Neurological", 
    "Cardiovascular",
    "Dermatological",
    "Musculoskeletal",
    "Psychiatric",
    "Respiratory",
    "Metabolic",
    "Allergic/Immunological",
    "Genitourinary",
    "Hematological",
    "Other"
]

# ============================================
# DASHBOARD SETTINGS
# ============================================
DASHBOARD_TITLE = "🏥 Pharma Vigilance Dashboard"
DASHBOARD_THEME = "dark"
PAGE_SIZE = 20  # Records per page in tables
