"""
============================================
PHASE 4: LLM-Based Adverse Event Extractor
============================================
Purpose: Use AI to extract structured adverse event data from patient reviews

What this module does:
1. Connects to Groq API (Llama 3) for high-speed inference
2. Uses carefully crafted prompts to extract medical entities
3. Validates and structures LLM outputs using Pydantic models
4. Implements "grounding" - requires exact quotes from source text
5. Handles rate limiting and batch processing

AI Approach Explained:
- Model: Llama 3.1 70B via Groq API (fast, cost-effective)
- Technique: Structured extraction with JSON output
- Validation: Pydantic models ensure data consistency
- Grounding: Each finding must cite exact source text

Why it's needed:
- Transforms unstructured patient language into analyzable data
- Understands context and nuance that keywords miss
- Standardizes diverse symptom descriptions
"""

import json
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

# Pydantic for structured output validation
from pydantic import BaseModel, Field, validator


# ============================================
# DATA MODELS (Structured Output Schemas)
# ============================================

class ExtractedSymptom(BaseModel):
    """
    Represents a single adverse event/symptom extracted from text.
    
    The 'source_quote' field is CRITICAL - it provides "grounding"
    by requiring the AI to cite exact text from the review.
    """
    symptom_name: str = Field(
        description="Standardized name of the symptom/adverse event"
    )
    severity: str = Field(
        description="Severity level: mild, moderate, severe, or life-threatening"
    )
    body_system: str = Field(
        description="Affected body system (e.g., gastrointestinal, neurological)"
    )
    duration: Optional[str] = Field(
        default=None,
        description="How long the symptom lasted (if mentioned)"
    )
    onset: Optional[str] = Field(
        default=None,
        description="When the symptom started relative to medication (if mentioned)"
    )
    source_quote: str = Field(
        description="EXACT quote from the review supporting this extraction"
    )
    
    @validator('severity')
    def normalize_severity(cls, v):
        """Normalize severity to standard levels."""
        v_lower = v.lower().strip()
        severity_mapping = {
            'mild': 'mild',
            'minor': 'mild',
            'slight': 'mild',
            'moderate': 'moderate',
            'medium': 'moderate',
            'severe': 'severe',
            'serious': 'severe',
            'bad': 'severe',
            'life-threatening': 'life-threatening',
            'life threatening': 'life-threatening',
            'emergency': 'life-threatening',
            'critical': 'life-threatening'
        }
        return severity_mapping.get(v_lower, 'moderate')
    
    @validator('body_system')
    def normalize_body_system(cls, v):
        """Normalize body system to standard categories."""
        v_lower = v.lower().strip()
        system_mapping = {
            'gastrointestinal': 'Gastrointestinal',
            'gi': 'Gastrointestinal',
            'stomach': 'Gastrointestinal',
            'digestive': 'Gastrointestinal',
            'neurological': 'Neurological',
            'nervous': 'Neurological',
            'brain': 'Neurological',
            'cardiovascular': 'Cardiovascular',
            'heart': 'Cardiovascular',
            'cardiac': 'Cardiovascular',
            'dermatological': 'Dermatological',
            'skin': 'Dermatological',
            'musculoskeletal': 'Musculoskeletal',
            'muscle': 'Musculoskeletal',
            'joint': 'Musculoskeletal',
            'psychiatric': 'Psychiatric',
            'mental': 'Psychiatric',
            'mood': 'Psychiatric',
            'respiratory': 'Respiratory',
            'lung': 'Respiratory',
            'breathing': 'Respiratory',
            'metabolic': 'Metabolic',
            'allergic': 'Allergic/Immunological',
            'immunological': 'Allergic/Immunological',
            'genitourinary': 'Genitourinary',
            'urinary': 'Genitourinary',
            'sexual': 'Genitourinary',
            'hematological': 'Hematological',
            'blood': 'Hematological'
        }
        for key, value in system_mapping.items():
            if key in v_lower:
                return value
        return 'Other'


class PatientAction(BaseModel):
    """Actions taken by the patient in response to side effects."""
    action_type: str = Field(
        description="Type of action: discontinued, reduced_dose, continued, sought_medical_help"
    )
    details: Optional[str] = Field(
        default=None,
        description="Additional details about the action"
    )
    source_quote: str = Field(
        description="EXACT quote from review supporting this"
    )


class ReviewExtraction(BaseModel):
    """
    Complete extraction result for a single patient review.
    """
    review_id: int = Field(description="Unique identifier for the review")
    drug_name: str = Field(description="Name of the medication")
    condition_treated: str = Field(description="Medical condition being treated")
    symptoms: List[ExtractedSymptom] = Field(
        default_factory=list,
        description="List of extracted adverse events/symptoms"
    )
    patient_actions: List[PatientAction] = Field(
        default_factory=list,
        description="Actions taken by patient"
    )
    overall_sentiment: str = Field(
        default="neutral",
        description="Overall sentiment: positive, negative, neutral, mixed"
    )
    extraction_confidence: float = Field(
        default=0.0,
        description="Confidence score 0-1 for this extraction"
    )


# ============================================
# PROMPT TEMPLATES
# ============================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert pharmacovigilance analyst specializing in extracting adverse drug reactions from patient reviews.

Your task is to carefully read patient medication reviews and extract structured information about:
1. Adverse events/side effects experienced
2. Severity of each symptom
3. Body systems affected
4. Duration and onset timing (if mentioned)
5. Actions taken by the patient

CRITICAL RULES:
- ONLY extract symptoms that are EXPLICITLY mentioned in the review
- For each symptom, you MUST provide an EXACT quote from the review as evidence
- If something is not mentioned, leave it as null - NEVER invent information
- Distinguish between symptoms caused by the medication vs. the underlying condition
- Use standardized medical terminology when possible

Severity Classification Guide:
- Mild: Minor discomfort, doesn't interfere with daily activities
- Moderate: Noticeable impact, may require intervention
- Severe: Significant impact, requires medical attention or causes major disruption
- Life-threatening: Requires immediate medical attention

Body Systems:
- Gastrointestinal (nausea, vomiting, diarrhea, constipation)
- Neurological (headache, dizziness, numbness, tingling)
- Cardiovascular (chest pain, palpitations, blood pressure changes)
- Dermatological (rash, itching, hives)
- Musculoskeletal (muscle pain, joint pain, weakness)
- Psychiatric (anxiety, depression, mood changes, insomnia)
- Respiratory (shortness of breath, cough)
- Metabolic (weight changes, appetite changes)
- Allergic/Immunological (allergic reactions)
- Genitourinary (urinary issues, sexual dysfunction)
- Hematological (bleeding, bruising)
- Other (anything not fitting above categories)

Output your response as a valid JSON object matching the specified schema."""


EXTRACTION_USER_PROMPT = """Analyze the following patient medication review and extract all adverse events/side effects.

DRUG: {drug_name}
CONDITION BEING TREATED: {condition}
PATIENT RATING: {rating}/10
REPORTED SIDE EFFECT SEVERITY: {side_effect_category}

REVIEW TEXT:
{review_text}

---

Extract all adverse events from this review. For each symptom found:
1. Provide the standardized symptom name
2. Classify its severity (mild/moderate/severe/life-threatening)
3. Identify the affected body system
4. Note duration and onset if mentioned
5. Include the EXACT quote from the review that mentions this symptom

Also identify any actions the patient took (discontinued medication, reduced dose, etc.)

Respond with a JSON object in this exact format:
{{
    "symptoms": [
        {{
            "symptom_name": "standardized symptom name",
            "severity": "mild|moderate|severe|life-threatening",
            "body_system": "affected body system",
            "duration": "duration if mentioned or null",
            "onset": "when it started if mentioned or null",
            "source_quote": "exact quote from review"
        }}
    ],
    "patient_actions": [
        {{
            "action_type": "discontinued|reduced_dose|continued|sought_medical_help",
            "details": "additional details or null",
            "source_quote": "exact quote from review"
        }}
    ],
    "overall_sentiment": "positive|negative|neutral|mixed",
    "extraction_confidence": 0.0-1.0
}}

If no adverse events are mentioned, return an empty symptoms array."""


# ============================================
# LLM EXTRACTOR CLASS
# ============================================

class AdverseEventExtractor:
    """
    Extracts structured adverse event data from patient reviews using LLMs.
    
    Usage:
        extractor = AdverseEventExtractor(api_key="your_groq_api_key")
        results = extractor.extract_from_dataframe(processed_df)
    """
    
    def __init__(
        self, 
        api_key: str = None,
        model: str = "llama-3.1-70b-versatile",
        use_groq: bool = True
    ):
        """
        Initialize the extractor.
        
        Args:
            api_key: Groq or OpenAI API key
            model: Model name to use
            use_groq: If True, use Groq; if False, use OpenAI
        """
        self.use_groq = use_groq
        self.model = model
        self.client = None
        
        # Get API key from config if not provided
        if api_key is None:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config import GROQ_API_KEY, OPENAI_API_KEY
            api_key = GROQ_API_KEY if use_groq else OPENAI_API_KEY
        
        if not api_key:
            print("⚠️ Warning: No API key provided. Set GROQ_API_KEY in your .env file")
            print("   Get a free key at: https://console.groq.com")
            return
        
        # Initialize the appropriate client
        if use_groq:
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
                print(f"✅ Groq client initialized with model: {model}")
            except ImportError:
                print("❌ Groq library not installed. Run: pip install groq")
        else:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                print(f"✅ OpenAI client initialized with model: {model}")
            except ImportError:
                print("❌ OpenAI library not installed. Run: pip install openai")
    
    def extract_single(self, review_data: Dict[str, Any]) -> ReviewExtraction:
        """
        Extract adverse events from a single review.
        
        Args:
            review_data: Dictionary with review information
            
        Returns:
            ReviewExtraction object with structured data
        """
        if self.client is None:
            return self._create_empty_extraction(review_data)
        
        # Build the prompt
        prompt = EXTRACTION_USER_PROMPT.format(
            drug_name=review_data.get('drug_name', 'Unknown'),
            condition=review_data.get('condition', 'Unknown'),
            rating=review_data.get('rating', 'N/A'),
            side_effect_category=review_data.get('side_effect_category', 'Unknown'),
            review_text=review_data.get('review_text', '')
        )
        
        try:
            # Make API call
            if self.use_groq:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent extraction
                    max_tokens=2000,
                    response_format={"type": "json_object"}
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                    response_format={"type": "json_object"}
                )
            
            # Parse response
            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)
            
            # Build ReviewExtraction object
            extraction = ReviewExtraction(
                review_id=review_data.get('review_id', 0),
                drug_name=review_data.get('drug_name', 'Unknown'),
                condition_treated=review_data.get('condition', 'Unknown'),
                symptoms=[
                    ExtractedSymptom(**s) for s in result_json.get('symptoms', [])
                ],
                patient_actions=[
                    PatientAction(**a) for a in result_json.get('patient_actions', [])
                ],
                overall_sentiment=result_json.get('overall_sentiment', 'neutral'),
                extraction_confidence=result_json.get('extraction_confidence', 0.5)
            )
            
            return extraction
            
        except Exception as e:
            print(f"❌ Extraction error: {str(e)}")
            return self._create_empty_extraction(review_data)
    
    def _create_empty_extraction(self, review_data: Dict) -> ReviewExtraction:
        """Create an empty extraction when API fails."""
        return ReviewExtraction(
            review_id=review_data.get('review_id', 0),
            drug_name=review_data.get('drug_name', 'Unknown'),
            condition_treated=review_data.get('condition', 'Unknown'),
            symptoms=[],
            patient_actions=[],
            overall_sentiment='unknown',
            extraction_confidence=0.0
        )
    
    def extract_batch(
        self, 
        batch: pd.DataFrame, 
        delay: float = 0.5
    ) -> List[ReviewExtraction]:
        """
        Extract adverse events from a batch of reviews.
        
        Args:
            batch: DataFrame with preprocessed reviews
            delay: Seconds between API calls (rate limiting)
            
        Returns:
            List of ReviewExtraction objects
        """
        results = []
        
        for idx, row in batch.iterrows():
            review_data = {
                'review_id': row.get('review_id', idx),
                'drug_name': row.get('urlDrugName', 'Unknown'),
                'condition': row.get('condition', 'Unknown'),
                'rating': row.get('rating', 'N/A'),
                'side_effect_category': row.get('sideEffects', 'Unknown'),
                'review_text': row.get('combined_review', row.get('sideEffectsReview', ''))
            }
            
            extraction = self.extract_single(review_data)
            results.append(extraction)
            
            # Rate limiting
            time.sleep(delay)
        
        return results
    
    def extract_from_dataframe(
        self, 
        df: pd.DataFrame,
        batch_size: int = 10,
        max_reviews: int = None,
        progress_callback=None
    ) -> pd.DataFrame:
        """
        Extract adverse events from an entire DataFrame.
        
        Args:
            df: Preprocessed DataFrame with reviews
            batch_size: Reviews per batch
            max_reviews: Maximum reviews to process (for testing)
            progress_callback: Function to call with progress updates
            
        Returns:
            DataFrame with extraction results
        """
        print("\n🤖 Starting AI extraction pipeline...")
        
        # Limit reviews if specified
        if max_reviews:
            df = df.head(max_reviews)
        
        total = len(df)
        all_results = []
        
        # Process in batches
        for i in range(0, total, batch_size):
            batch = df.iloc[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size
            
            print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} reviews)...")
            
            results = self.extract_batch(batch)
            all_results.extend(results)
            
            if progress_callback:
                progress_callback(i + len(batch), total)
        
        # Convert to DataFrame
        results_df = self._extractions_to_dataframe(all_results)
        
        print(f"\n✅ Extraction complete!")
        print(f"   - Processed: {len(all_results)} reviews")
        print(f"   - Total symptoms found: {len(results_df)}")
        
        return results_df
    
    def _extractions_to_dataframe(self, extractions: List[ReviewExtraction]) -> pd.DataFrame:
        """
        Convert list of extractions to a flat DataFrame.
        
        Each row represents one symptom, linked to its source review.
        """
        rows = []
        
        for extraction in extractions:
            for symptom in extraction.symptoms:
                rows.append({
                    'review_id': extraction.review_id,
                    'drug_name': extraction.drug_name,
                    'condition_treated': extraction.condition_treated,
                    'symptom_name': symptom.symptom_name,
                    'severity': symptom.severity,
                    'body_system': symptom.body_system,
                    'duration': symptom.duration,
                    'onset': symptom.onset,
                    'source_quote': symptom.source_quote,
                    'overall_sentiment': extraction.overall_sentiment,
                    'extraction_confidence': extraction.extraction_confidence
                })
        
        return pd.DataFrame(rows)


# ============================================
# MOCK EXTRACTOR (For testing without API)
# ============================================

class MockExtractor:
    """
    Mock extractor for testing without API access.
    Simulates extraction using keyword matching.
    """
    
    # Common symptom keywords and their mappings
    SYMPTOM_KEYWORDS = {
        'nausea': ('Nausea', 'moderate', 'Gastrointestinal'),
        'vomiting': ('Vomiting', 'moderate', 'Gastrointestinal'),
        'diarrhea': ('Diarrhea', 'moderate', 'Gastrointestinal'),
        'constipation': ('Constipation', 'mild', 'Gastrointestinal'),
        'headache': ('Headache', 'mild', 'Neurological'),
        'dizziness': ('Dizziness', 'moderate', 'Neurological'),
        'dizzy': ('Dizziness', 'moderate', 'Neurological'),
        'drowsiness': ('Drowsiness', 'mild', 'Neurological'),
        'drowsy': ('Drowsiness', 'mild', 'Neurological'),
        'fatigue': ('Fatigue', 'mild', 'Neurological'),
        'tired': ('Fatigue', 'mild', 'Neurological'),
        'insomnia': ('Insomnia', 'moderate', 'Psychiatric'),
        'anxiety': ('Anxiety', 'moderate', 'Psychiatric'),
        'depression': ('Depression', 'severe', 'Psychiatric'),
        'rash': ('Skin Rash', 'moderate', 'Dermatological'),
        'itching': ('Pruritus', 'mild', 'Dermatological'),
        'weight gain': ('Weight Gain', 'moderate', 'Metabolic'),
        'weight loss': ('Weight Loss', 'moderate', 'Metabolic'),
        'dry mouth': ('Xerostomia', 'mild', 'Gastrointestinal'),
        'sweating': ('Hyperhidrosis', 'mild', 'Dermatological'),
    }
    
    def extract_from_dataframe(
        self, 
        df: pd.DataFrame,
        max_reviews: int = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Mock extraction using keyword matching.
        """
        print("\n🔧 Running MOCK extraction (no API key provided)...")
        
        if max_reviews:
            df = df.head(max_reviews)
        
        rows = []
        
        for idx, row in df.iterrows():
            text = str(row.get('combined_review', row.get('sideEffectsReview', ''))).lower()
            
            for keyword, (symptom, severity, system) in self.SYMPTOM_KEYWORDS.items():
                if keyword in text:
                    # Find the context around the keyword
                    start = max(0, text.find(keyword) - 30)
                    end = min(len(text), text.find(keyword) + len(keyword) + 30)
                    quote = text[start:end].strip()
                    
                    rows.append({
                        'review_id': row.get('review_id', idx),
                        'drug_name': row.get('urlDrugName', 'Unknown'),
                        'condition_treated': row.get('condition', 'Unknown'),
                        'symptom_name': symptom,
                        'severity': severity,
                        'body_system': system,
                        'duration': None,
                        'onset': None,
                        'source_quote': f"...{quote}...",
                        'overall_sentiment': 'extracted_mock',
                        'extraction_confidence': 0.5
                    })
        
        print(f"✅ Mock extraction complete: {len(rows)} symptoms found")
        return pd.DataFrame(rows)


def get_extractor(api_key: str = None) -> Any:
    """
    Factory function to get the appropriate extractor.
    Returns MockExtractor if no API key is available.
    """
    if api_key:
        return AdverseEventExtractor(api_key=api_key)
    
    # Try to get from config
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from config import GROQ_API_KEY
        if GROQ_API_KEY:
            return AdverseEventExtractor(api_key=GROQ_API_KEY)
    except:
        pass
    
    print("⚠️ No API key found. Using MockExtractor for demonstration.")
    return MockExtractor()


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    """
    Test the extractor by running:
    python src/llm_extractor.py
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.data_loader import DrugReviewLoader
    from src.preprocessor import DrugReviewPreprocessor
    
    print("\n" + "🤖 "*20)
    print("PHARMA VIGILANCE - LLM Extraction Module")
    print("🤖 "*20 + "\n")
    
    # Load and preprocess data
    loader = DrugReviewLoader()
    train_df, _ = loader.load_all_data()
    
    preprocessor = DrugReviewPreprocessor()
    processed_df = preprocessor.preprocess(train_df)
    
    # Get extractor (will use mock if no API key)
    extractor = get_extractor()
    
    # Extract from a small sample
    print("\n📊 Testing extraction on 20 reviews...")
    results_df = extractor.extract_from_dataframe(
        processed_df,
        max_reviews=20,
        batch_size=5
    )
    
    # Display results
    print("\n" + "="*60)
    print("📋 EXTRACTION RESULTS")
    print("="*60)
    
    if len(results_df) > 0:
        print(f"\nTotal symptoms extracted: {len(results_df)}")
        
        print("\n📊 Symptoms by Body System:")
        print(results_df['body_system'].value_counts().to_string())
        
        print("\n📊 Symptoms by Severity:")
        print(results_df['severity'].value_counts().to_string())
        
        print("\n📋 Sample Extractions:")
        for _, row in results_df.head(5).iterrows():
            print(f"\n   💊 Drug: {row['drug_name']}")
            print(f"   🩺 Symptom: {row['symptom_name']} ({row['severity']})")
            print(f"   🏥 System: {row['body_system']}")
            print(f"   📝 Quote: \"{row['source_quote'][:80]}...\"")
    else:
        print("No symptoms extracted. Check your API key or review data.")

