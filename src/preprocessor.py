"""
============================================
PHASE 3: Data Preprocessing Module
============================================
Purpose: Clean and prepare data for AI processing

What this module does:
1. Cleans text data (remove noise, normalize formatting)
2. Anonymizes potentially identifying information
3. Combines relevant text fields for analysis
4. Filters and validates data quality
5. Prepares batches for LLM processing

Why it's needed:
- Raw patient reviews contain noise and formatting issues
- Privacy protection is critical before cloud API processing
- Clean data improves AI extraction accuracy
- Batching enables efficient API usage
"""

import pandas as pd
import numpy as np
import re
from typing import List, Optional, Tuple
from pathlib import Path


class TextPreprocessor:
    """
    Cleans and prepares text data for AI analysis.
    
    Usage:
        preprocessor = TextPreprocessor()
        cleaned_df = preprocessor.clean_dataframe(raw_df)
    """
    
    def __init__(self):
        """Initialize the preprocessor with regex patterns."""
        
        # Patterns for anonymization (privacy protection)
        self.anonymization_patterns = {
            # Email addresses
            'email': (
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                '[EMAIL_REMOVED]'
            ),
            # Phone numbers (various formats)
            'phone': (
                r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
                '[PHONE_REMOVED]'
            ),
            # Social Security Numbers
            'ssn': (
                r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
                '[SSN_REMOVED]'
            ),
            # Names after common identifiers (Dr., Dr, Mr., Mrs., etc.)
            'names_with_title': (
                r'\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Miss)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',
                '[NAME_REMOVED]'
            ),
            # URLs
            'urls': (
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                '[URL_REMOVED]'
            )
        }
        
        # Patterns for text cleaning
        self.cleaning_patterns = [
            # Multiple whitespace to single space
            (r'\s+', ' '),
            # Multiple newlines to single newline
            (r'\n+', '\n'),
            # Remove leading/trailing quotes that wrap entire text
            (r'^["\'](.*)["\']$', r'\1'),
        ]
    
    def anonymize_text(self, text: str) -> str:
        """
        Remove potentially identifying information from text.
        
        Args:
            text: Raw text that may contain PII
            
        Returns:
            Text with PII removed/replaced
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        anonymized = text
        for pattern_name, (pattern, replacement) in self.anonymization_patterns.items():
            anonymized = re.sub(pattern, replacement, anonymized, flags=re.IGNORECASE)
        
        return anonymized
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text formatting.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        cleaned = text
        
        # Apply cleaning patterns
        for pattern, replacement in self.cleaning_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Strip whitespace
        cleaned = cleaned.strip()
        
        # Remove excessive special characters
        cleaned = re.sub(r'[^\w\s.,!?;:\'\"-]', '', cleaned)
        
        return cleaned
    
    def process_text(self, text: str, anonymize: bool = True) -> str:
        """
        Full text processing pipeline.
        
        Args:
            text: Raw text
            anonymize: Whether to remove PII
            
        Returns:
            Processed text
        """
        if anonymize:
            text = self.anonymize_text(text)
        return self.clean_text(text)


class DrugReviewPreprocessor:
    """
    Preprocesses drug review DataFrames for AI analysis.
    
    Usage:
        preprocessor = DrugReviewPreprocessor()
        cleaned_df = preprocessor.preprocess(raw_df)
        batches = preprocessor.create_batches(cleaned_df, batch_size=10)
    """
    
    def __init__(self, anonymize: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            anonymize: Whether to remove PII from reviews
        """
        self.text_processor = TextPreprocessor()
        self.anonymize = anonymize
        
        # Columns containing text that needs processing
        self.text_columns = ['benefitsReview', 'sideEffectsReview', 'commentsReview']
        
        # Effectiveness level mapping (for numeric analysis)
        self.effectiveness_map = {
            'Highly Effective': 5,
            'Considerably Effective': 4,
            'Moderately Effective': 3,
            'Marginally Effective': 2,
            'Ineffective': 1
        }
        
        # Side effects severity mapping
        self.severity_map = {
            'No Side Effects': 0,
            'Mild Side Effects': 1,
            'Moderate Side Effects': 2,
            'Severe Side Effects': 3,
            'Extremely Severe Side Effects': 4
        }
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full preprocessing pipeline for drug reviews.
        
        Args:
            df: Raw DataFrame with drug reviews
            
        Returns:
            Cleaned and preprocessed DataFrame
        """
        print("🔧 Starting preprocessing pipeline...")
        
        # Create a copy to avoid modifying original
        processed = df.copy()
        
        # Step 1: Clean text columns
        print("   📝 Cleaning text columns...")
        for col in self.text_columns:
            if col in processed.columns:
                processed[col] = processed[col].apply(
                    lambda x: self.text_processor.process_text(x, self.anonymize)
                )
        
        # Step 2: Normalize drug names (lowercase, strip whitespace)
        print("   💊 Normalizing drug names...")
        if 'urlDrugName' in processed.columns:
            processed['drug_name_clean'] = (
                processed['urlDrugName']
                .str.lower()
                .str.strip()
                .str.replace(r'[^a-z0-9\s-]', '', regex=True)
            )
        
        # Step 3: Clean condition names
        print("   🏥 Normalizing conditions...")
        if 'condition' in processed.columns:
            processed['condition_clean'] = (
                processed['condition']
                .str.lower()
                .str.strip()
            )
        
        # Step 4: Create numeric effectiveness score
        print("   📊 Creating numeric scores...")
        if 'effectiveness' in processed.columns:
            processed['effectiveness_score'] = (
                processed['effectiveness'].map(self.effectiveness_map)
            )
        
        # Step 5: Create numeric severity score
        if 'sideEffects' in processed.columns:
            processed['severity_score'] = (
                processed['sideEffects'].map(self.severity_map)
            )
        
        # Step 6: Combine text fields for comprehensive analysis
        print("   🔗 Creating combined review field...")
        processed['combined_review'] = self._combine_reviews(processed)
        
        # Step 7: Calculate text lengths (useful for filtering)
        processed['review_length'] = processed['combined_review'].str.len()
        
        # Step 8: Filter out empty or too-short reviews
        print("   🔍 Filtering low-quality records...")
        min_length = 20  # Minimum characters for meaningful review
        original_count = len(processed)
        processed = processed[processed['review_length'] >= min_length]
        filtered_count = original_count - len(processed)
        print(f"      Removed {filtered_count:,} records with insufficient text")
        
        # Step 9: Add unique identifier
        processed['review_id'] = range(1, len(processed) + 1)
        
        print(f"✅ Preprocessing complete! {len(processed):,} records ready for analysis")
        
        return processed
    
    def _combine_reviews(self, df: pd.DataFrame) -> pd.Series:
        """
        Combine all review text fields into one comprehensive field.
        
        Args:
            df: DataFrame with text columns
            
        Returns:
            Series with combined text
        """
        combined = []
        
        for _, row in df.iterrows():
            parts = []
            
            # Add benefits review
            if 'benefitsReview' in df.columns and pd.notna(row.get('benefitsReview')):
                benefits = str(row['benefitsReview']).strip()
                if benefits:
                    parts.append(f"Benefits: {benefits}")
            
            # Add side effects review (most important for pharmacovigilance)
            if 'sideEffectsReview' in df.columns and pd.notna(row.get('sideEffectsReview')):
                side_effects = str(row['sideEffectsReview']).strip()
                if side_effects:
                    parts.append(f"Side Effects: {side_effects}")
            
            # Add additional comments
            if 'commentsReview' in df.columns and pd.notna(row.get('commentsReview')):
                comments = str(row['commentsReview']).strip()
                if comments:
                    parts.append(f"Comments: {comments}")
            
            combined.append('\n'.join(parts))
        
        return pd.Series(combined, index=df.index)
    
    def create_batches(
        self, 
        df: pd.DataFrame, 
        batch_size: int = 10
    ) -> List[pd.DataFrame]:
        """
        Split DataFrame into batches for API processing.
        
        Args:
            df: Preprocessed DataFrame
            batch_size: Number of records per batch
            
        Returns:
            List of DataFrame batches
        """
        batches = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            batches.append(batch)
        
        print(f"📦 Created {len(batches)} batches of ~{batch_size} records each")
        return batches
    
    def get_processing_stats(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict:
        """
        Calculate preprocessing statistics.
        
        Args:
            original_df: Raw DataFrame before preprocessing
            processed_df: DataFrame after preprocessing
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'original_records': len(original_df),
            'processed_records': len(processed_df),
            'records_removed': len(original_df) - len(processed_df),
            'removal_rate': (len(original_df) - len(processed_df)) / len(original_df) * 100,
            'avg_review_length': processed_df['review_length'].mean(),
            'total_text_volume': processed_df['review_length'].sum(),
            'unique_drugs': processed_df['drug_name_clean'].nunique() if 'drug_name_clean' in processed_df.columns else 0,
            'unique_conditions': processed_df['condition_clean'].nunique() if 'condition_clean' in processed_df.columns else 0
        }
        
        return stats


def create_analysis_sample(
    df: pd.DataFrame, 
    n_samples: int = 100, 
    stratify_by: str = 'sideEffects'
) -> pd.DataFrame:
    """
    Create a stratified sample for testing/development.
    
    Useful for testing the pipeline without processing all data.
    
    Args:
        df: Full preprocessed DataFrame
        n_samples: Total number of samples
        stratify_by: Column to stratify by
        
    Returns:
        Stratified sample DataFrame
    """
    if stratify_by not in df.columns:
        return df.sample(n=min(n_samples, len(df)))
    
    # Calculate samples per stratum
    value_counts = df[stratify_by].value_counts()
    samples_per_group = n_samples // len(value_counts)
    
    samples = []
    for value in value_counts.index:
        group = df[df[stratify_by] == value]
        n = min(samples_per_group, len(group))
        samples.append(group.sample(n=n))
    
    result = pd.concat(samples)
    print(f"📊 Created stratified sample of {len(result)} records")
    return result


# ============================================
# MAIN EXECUTION - Run this file directly
# ============================================
if __name__ == "__main__":
    """
    Test the preprocessor by running:
    python src/preprocessor.py
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.data_loader import DrugReviewLoader
    
    print("\n" + "🔧 "*20)
    print("PHARMA VIGILANCE - Preprocessing Module")
    print("🔧 "*20 + "\n")
    
    # Load data
    loader = DrugReviewLoader()
    train_df, _ = loader.load_all_data()
    
    # Preprocess
    preprocessor = DrugReviewPreprocessor(anonymize=True)
    processed_df = preprocessor.preprocess(train_df)
    
    # Show statistics
    stats = preprocessor.get_processing_stats(train_df, processed_df)
    print("\n📊 Preprocessing Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value:,}")
    
    # Show sample processed record
    print("\n" + "="*60)
    print("📝 SAMPLE PROCESSED RECORD")
    print("="*60)
    
    sample = processed_df.sample(1).iloc[0]
    print(f"\n💊 Drug: {sample['urlDrugName']}")
    print(f"🏥 Condition: {sample['condition']}")
    print(f"⭐ Rating: {sample['rating']}")
    print(f"📏 Review Length: {sample['review_length']} characters")
    print(f"\n📖 Combined Review:\n{sample['combined_review'][:500]}...")
    
    # Create batches
    batches = preprocessor.create_batches(processed_df.head(100), batch_size=10)
    print(f"\n✅ Ready for LLM processing: {len(batches)} batches prepared")

