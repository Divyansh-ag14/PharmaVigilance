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
# API CONFIGURATION
# ============================================
# Get API key from environment variable
# Create a .env file with: GROQ_API_KEY=your_key_here
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Model selection for Groq
# Options: "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"
LLM_MODEL = "llama-3.1-70b-versatile"

# Optional: OpenAI for production
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4-turbo-preview"

# ============================================
# PROCESSING SETTINGS
# ============================================
# Batch size for LLM processing (to manage rate limits)
BATCH_SIZE = 10

# Maximum reviews to process (for testing, set to None for all)
MAX_REVIEWS = None

# Rate limiting (requests per minute)
RATE_LIMIT_RPM = 30

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

