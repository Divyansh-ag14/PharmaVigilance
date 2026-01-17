"""
change
============================================
PHASE 2: Data Loader Module
============================================
Purpose: Load and explore the UCI Drug Review Dataset

What this module does:
1. Loads TSV data files into pandas DataFrames
2. Provides initial data exploration and statistics
3. Demonstrates why keyword-based approaches fall short
4. Offers utility functions for data access

Why it's needed:
- Establishes the data foundation for all downstream processing
- Helps understand the data structure before applying AI
- Identifies data quality issues early
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import re


class DrugReviewLoader:
    """
    Loads and manages drug review data from TSV files.
    
    Usage:
        loader = DrugReviewLoader()
        train_df, test_df = loader.load_all_data()
        loader.show_data_summary(train_df)
    """
    
    def __init__(self, train_path: str = None, test_path: str = None):
        """
        Initialize the data loader with file paths.
        
        Args:
            train_path: Path to training data TSV file
            test_path: Path to test data TSV file
        """
        # Import config for default paths
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import TRAIN_DATA_PATH, TEST_DATA_PATH
        
        self.train_path = Path(train_path) if train_path else TRAIN_DATA_PATH
        self.test_path = Path(test_path) if test_path else TEST_DATA_PATH
        
        # Expected columns in the dataset
        self.expected_columns = [
            'urlDrugName',      # Name of the drug
            'rating',           # Patient rating (1-10)
            'effectiveness',    # How effective (categorical)
            'sideEffects',      # Side effect severity (categorical)
            'condition',        # Medical condition treated
            'benefitsReview',   # Text review of benefits
            'sideEffectsReview', # Text review of side effects
            'commentsReview'    # Additional comments
        ]
    
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single TSV file into a DataFrame.
        
        Args:
            file_path: Path to the TSV file
            
        Returns:
            DataFrame with the loaded data
        """
        print(f"📂 Loading data from: {file_path.name}")
        
        # Load TSV file (tab-separated values)
        df = pd.read_csv(
            file_path,
            sep='\t',           # Tab separator
            encoding='utf-8',   # Handle special characters
            on_bad_lines='skip' # Skip malformed rows
        )
        
        # The first column might be an index, let's handle that
        if df.columns[0] == '' or df.columns[0].isdigit():
            df = df.iloc[:, 1:]  # Remove the index column
        
        print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
        return df
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both training and test datasets.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        train_df = self.load_data(self.train_path)
        test_df = self.load_data(self.test_path)
        
        print(f"\n📊 Total records: {len(train_df) + len(test_df):,}")
        print(f"   - Training: {len(train_df):,}")
        print(f"   - Test: {len(test_df):,}")
        
        return train_df, test_df
    
    def show_data_summary(self, df: pd.DataFrame, name: str = "Dataset") -> Dict:
        """
        Display comprehensive summary statistics.
        
        Args:
            df: DataFrame to analyze
            name: Name for display purposes
            
        Returns:
            Dictionary with summary statistics
        """
        print(f"\n{'='*60}")
        print(f"📈 {name} Summary")
        print(f"{'='*60}")
        
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'unique_drugs': df['urlDrugName'].nunique() if 'urlDrugName' in df.columns else 0,
            'unique_conditions': df['condition'].nunique() if 'condition' in df.columns else 0
        }
        
        print(f"\n📋 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"\n📝 Columns: {', '.join(df.columns)}")
        
        # Missing values
        print(f"\n❓ Missing Values:")
        for col, count in summary['missing_values'].items():
            pct = (count / len(df)) * 100
            print(f"   - {col}: {count:,} ({pct:.1f}%)")
        
        # Unique counts
        print(f"\n💊 Unique Drugs: {summary['unique_drugs']:,}")
        print(f"🏥 Unique Conditions: {summary['unique_conditions']:,}")
        
        # Rating distribution
        if 'rating' in df.columns:
            print(f"\n⭐ Rating Distribution:")
            rating_dist = df['rating'].value_counts().sort_index()
            for rating, count in rating_dist.items():
                bar = '█' * int(count / len(df) * 50)
                print(f"   {rating:2}: {bar} ({count:,})")
        
        # Effectiveness distribution
        if 'effectiveness' in df.columns:
            print(f"\n💪 Effectiveness Distribution:")
            for eff, count in df['effectiveness'].value_counts().items():
                pct = (count / len(df)) * 100
                print(f"   - {eff}: {count:,} ({pct:.1f}%)")
        
        # Side effects severity distribution
        if 'sideEffects' in df.columns:
            print(f"\n⚠️ Side Effects Severity:")
            for sev, count in df['sideEffects'].value_counts().items():
                pct = (count / len(df)) * 100
                print(f"   - {sev}: {count:,} ({pct:.1f}%)")
        
        return summary
    
    def get_sample_reviews(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Get random sample reviews for inspection.
        
        Args:
            df: DataFrame to sample from
            n: Number of samples
            
        Returns:
            DataFrame with sample reviews
        """
        return df.sample(n=min(n, len(df)))
    
    def get_reviews_by_drug(self, df: pd.DataFrame, drug_name: str) -> pd.DataFrame:
        """
        Get all reviews for a specific drug.
        
        Args:
            df: DataFrame to search
            drug_name: Name of the drug (case-insensitive)
            
        Returns:
            DataFrame with matching reviews
        """
        mask = df['urlDrugName'].str.lower() == drug_name.lower()
        return df[mask]
    
    def get_reviews_by_condition(self, df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """
        Get all reviews for a specific condition.
        
        Args:
            df: DataFrame to search
            condition: Medical condition (partial match, case-insensitive)
            
        Returns:
            DataFrame with matching reviews
        """
        mask = df['condition'].str.lower().str.contains(condition.lower(), na=False)
        return df[mask]


def demonstrate_keyword_limitations(df: pd.DataFrame) -> None:
    """
    Demonstrates why simple keyword matching is insufficient for
    extracting adverse events from patient reviews.
    
    This is a KEY DELIVERABLE from Phase 1 - showing the need for AI.
    """
    print("\n" + "="*70)
    print("🔍 DEMONSTRATING KEYWORD APPROACH LIMITATIONS")
    print("="*70)
    
    # Example: Looking for "dizziness" related symptoms
    keyword = "dizziness"
    
    # Simple keyword match
    simple_matches = df[
        df['sideEffectsReview'].str.lower().str.contains(keyword, na=False)
    ]
    
    print(f"\n1️⃣ Simple keyword search for '{keyword}':")
    print(f"   Found: {len(simple_matches):,} reviews")
    
    # Alternative expressions patients use for dizziness
    dizziness_variants = [
        "dizzy", "dizziness", "lightheaded", "light-headed",
        "room spinning", "vertigo", "unsteady", "off balance",
        "wobbly", "woozy", "faint", "head rush", "spinning sensation"
    ]
    
    # Create regex pattern for all variants
    pattern = '|'.join(dizziness_variants)
    expanded_matches = df[
        df['sideEffectsReview'].str.lower().str.contains(pattern, na=False, regex=True)
    ]
    
    print(f"\n2️⃣ Expanded keyword search (13 variants):")
    print(f"   Found: {len(expanded_matches):,} reviews")
    print(f"   📈 Improvement: +{len(expanded_matches) - len(simple_matches):,} reviews found")
    
    # Show examples that would be MISSED by simple keyword search
    print(f"\n3️⃣ Examples that SIMPLE SEARCH MISSED:")
    missed = expanded_matches[
        ~expanded_matches['sideEffectsReview'].str.lower().str.contains(keyword, na=False)
    ].head(3)
    
    for idx, row in missed.iterrows():
        review = row['sideEffectsReview'][:200] if pd.notna(row['sideEffectsReview']) else "N/A"
        print(f"\n   Drug: {row['urlDrugName']}")
        print(f"   Review: \"{review}...\"")
    
    # Show what even EXPANDED keywords miss
    print(f"\n4️⃣ What even EXPANDED keywords might miss:")
    examples = [
        "I felt like the room was moving around me",
        "Everything went dark for a second when I stood up",
        "My balance was completely off",
        "I couldn't walk straight after taking it"
    ]
    for example in examples:
        matched = any(v in example.lower() for v in dizziness_variants)
        status = "✅ Caught" if matched else "❌ Missed"
        print(f"   {status}: \"{example}\"")
    
    print(f"\n💡 CONCLUSION:")
    print(f"   Keyword approaches require manual maintenance of extensive term lists")
    print(f"   and still miss nuanced descriptions. AI models understand CONTEXT")
    print(f"   and can identify symptoms regardless of how patients describe them.")


# ============================================
# MAIN EXECUTION - Run this file directly
# ============================================
if __name__ == "__main__":
    """
    Test the data loader by running this file directly:
    python src/data_loader.py
    """
    print("\n" + "🏥 "*20)
    print("PHARMA VIGILANCE - Data Loading Module")
    print("🏥 "*20 + "\n")
    
    # Initialize loader
    loader = DrugReviewLoader()
    
    # Load data
    train_df, test_df = loader.load_all_data()
    
    # Show summary
    loader.show_data_summary(train_df, "Training Data")
    
    # Demonstrate keyword limitations
    demonstrate_keyword_limitations(train_df)
    
    # Show sample reviews
    print("\n" + "="*60)
    print("📝 SAMPLE REVIEWS")
    print("="*60)
    
    samples = loader.get_sample_reviews(train_df, n=3)
    for idx, row in samples.iterrows():
        print(f"\n{'─'*50}")
        print(f"💊 Drug: {row['urlDrugName']}")
        print(f"🏥 Condition: {row['condition']}")
        print(f"⭐ Rating: {row['rating']}/10")
        print(f"⚠️ Side Effects Severity: {row['sideEffects']}")
        print(f"\n📖 Side Effects Review:")
        review = row['sideEffectsReview']
        if pd.notna(review):
            print(f"   {review[:300]}...")

