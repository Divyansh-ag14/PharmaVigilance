"""
============================================
PHASE 4: LLM-Based Review Structuring & Analysis
============================================
Purpose: Use AI to extract structured clinical data from raw patient reviews

Improved with:
- Robust retry logic with exponential backoff
- Timeout handling
- Connection error recovery
- Fallback to faster models
- Progress tracking
"""

import json
import time
import httpx
from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import re

# Pydantic for structured output validation
from pydantic import BaseModel, Field, field_validator


# ============================================
# DATA MODELS (Structured Output Schemas)
# ============================================

class StructuredReview(BaseModel):
    """Structured patient review matching dataset schema."""
    urlDrugName: str = Field(description="Normalized drug name")
    condition: str = Field(description="Medical condition being treated")
    rating: int = Field(ge=1, le=10, description="Patient rating from 1-10")
    effectiveness: str = Field(description="Effectiveness level (5 categories)")
    sideEffects: str = Field(description="Side effects severity (5 categories)")
    benefitsReview: Optional[str] = Field(default=None)
    sideEffectsReview: Optional[str] = Field(default=None)
    commentsReview: Optional[str] = Field(default=None)
    
    @field_validator('effectiveness')
    @classmethod
    def validate_effectiveness(cls, v):
        valid_levels = [
            "Highly Effective", "Considerably Effective", 
            "Moderately Effective", "Marginally Effective", "Ineffective"
        ]
        v_normalized = v.strip()
        for level in valid_levels:
            if level.lower() in v_normalized.lower():
                return level
        v_lower = v.lower()
        if any(word in v_lower for word in ['highly', 'excellent', 'amazing', 'perfect']):
            return "Highly Effective"
        elif any(word in v_lower for word in ['considerably', 'very', 'great']):
            return "Considerably Effective"
        elif any(word in v_lower for word in ['moderate', 'okay', 'decent']):
            return "Moderately Effective"
        elif any(word in v_lower for word in ['marginal', 'slight', 'barely']):
            return "Marginally Effective"
        elif any(word in v_lower for word in ['ineffective', 'not', 'didn\'t', 'worse']):
            return "Ineffective"
        return "Moderately Effective"
    
    @field_validator('sideEffects')
    @classmethod
    def validate_side_effects(cls, v):
        valid_levels = [
            "No Side Effects", "Mild Side Effects", "Moderate Side Effects", 
            "Severe Side Effects", "Extremely Severe Side Effects"
        ]
        v_normalized = v.strip()
        for level in valid_levels:
            if level.lower() in v_normalized.lower():
                return level
        v_lower = v.lower()
        if 'no ' in v_lower or 'none' in v_lower:
            return "No Side Effects"
        elif 'extremely' in v_lower or 'hospital' in v_lower:
            return "Extremely Severe Side Effects"
        elif 'severe' in v_lower or 'serious' in v_lower:
            return "Severe Side Effects"
        elif 'moderate' in v_lower:
            return "Moderate Side Effects"
        elif 'mild' in v_lower or 'minor' in v_lower:
            return "Mild Side Effects"
        return "Moderate Side Effects"
    
    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v):
        if isinstance(v, str):
            try:
                v = int(float(v))
            except:
                v = 5
        return max(1, min(10, v))


# ============================================
# PROMPT TEMPLATES
# ============================================

EXTRACTION_SYSTEM_PROMPT = """You are a Medical Pharmacovigilance Analyst. Extract structured data from drug reviews into JSON.

EXTRACTION RULES:
1. urlDrugName: Normalize drug name (e.g., "xanax" -> "Xanax")
2. condition: Medical reason for taking drug. Infer if not stated.
3. rating (1-10): Extract if present, otherwise infer from sentiment:
   - 9-10: "Miracle", "Life-changing", "Perfect"
   - 7-8: "Great", "Very effective"
   - 5-6: "Okay", "Average"
   - 3-4: "Disappointing", "Not great"
   - 1-2: "Horrible", "Dangerous", "Never again"

4. effectiveness - ONE of:
   - "Highly Effective"
   - "Considerably Effective"
   - "Moderately Effective"
   - "Marginally Effective"
   - "Ineffective"

5. sideEffects - ONE of:
   - "No Side Effects"
   - "Mild Side Effects"
   - "Moderate Side Effects"
   - "Severe Side Effects"
   - "Extremely Severe Side Effects"

6. benefitsReview: Extract positive effects text
7. sideEffectsReview: Extract adverse effects text
8. commentsReview: Extract dosage, context, conclusions

Return ONLY valid JSON, no explanation."""

EXTRACTION_USER_PROMPT = """Extract structured data from this drug review:

{review_text}

Return JSON:
{{"urlDrugName": "string", "condition": "string", "rating": int, "effectiveness": "string", "sideEffects": "string", "benefitsReview": "string or null", "sideEffectsReview": "string or null", "commentsReview": "string or null"}}"""


# ============================================
# ROBUST API CLIENT
# ============================================

class RobustAPIClient:
    """
    Robust API client with retry logic, timeout handling, and fallback models.
    """
    
    def __init__(
        self, 
        api_key: str = None, 
        model: str = None,
        timeout: int = 60,
        max_retries: int = 3
    ):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import UF_API_BASE, UF_API_KEY, LLM_MODEL
        
        self.api_base = UF_API_BASE
        self.api_key = api_key or UF_API_KEY
        self.model = model or LLM_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.initialized = False
        
        # Fallback models (faster/smaller)
        self.fallback_models = [
            "llama-3.1-8b-instruct",
            "mistral-7b-instruct",
        ]
        
        if not self.api_key:
            print("⚠️ Warning: No API key provided.")
            return
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if API is reachable."""
        try:
            # Quick connectivity test
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{self.api_base}/models", 
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                if response.status_code in [200, 401, 404]:
                    # API is reachable (even 401/404 means server responded)
                    self.initialized = True
                    print(f"✅ API connection verified")
                    print(f"   Endpoint: {self.api_base}")
                    print(f"   Model: {self.model}")
        except Exception as e:
            print(f"⚠️ Could not verify API connection: {e}")
            self.initialized = True  # Try anyway
    
    def invoke(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Make API request with retry logic and timeout handling.
        """
        if not self.api_key:
            return None
        
        models_to_try = [self.model] + self.fallback_models
        
        for model in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    result = self._make_request(model, system_prompt, user_prompt)
                    if result:
                        return result
                except httpx.TimeoutException:
                    wait_time = (attempt + 1) * 2
                    print(f"   ⏱️ Timeout (attempt {attempt + 1}/{self.max_retries}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                except httpx.ConnectError as e:
                    print(f"   🔌 Connection error: {e}")
                    time.sleep(2)
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    time.sleep(1)
            
            # Try next model
            if model != models_to_try[-1]:
                print(f"   🔄 Trying fallback model...")
        
        return None
    
    def _make_request(self, model: str, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Make a single API request."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1500
        }
        
        with httpx.Client(timeout=httpx.Timeout(self.timeout)) as client:
            response = client.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                print(f"   ⚠️ API returned {response.status_code}: {response.text[:100]}")
                return None


# ============================================
# REVIEW STRUCTURER
# ============================================

class ReviewStructurer:
    """Structures raw patient reviews using LLM."""
    
    def __init__(self, api_key: str = None, model: str = None, timeout: int = 60):
        self.client = RobustAPIClient(api_key=api_key, model=model, timeout=timeout)
    
    def structure_single(self, review_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Structure a single review."""
        
        if not self.client.initialized:
            return self._create_empty_result(review_text)
        
        # Add context
        enriched_text = review_text
        if context:
            if context.get('drug_name'):
                enriched_text = f"Drug: {context['drug_name']}\n{enriched_text}"
            if context.get('condition'):
                enriched_text = f"Condition: {context['condition']}\n{enriched_text}"
        
        # Truncate if too long
        if len(enriched_text) > 2000:
            enriched_text = enriched_text[:2000] + "..."
        
        user_prompt = EXTRACTION_USER_PROMPT.format(review_text=enriched_text)
        
        try:
            response_text = self.client.invoke(EXTRACTION_SYSTEM_PROMPT, user_prompt)
            
            if not response_text:
                return self._create_empty_result(review_text)
            
            # Extract JSON
            json_text = response_text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0]
            
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*\}', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()
            
            result_json = json.loads(json_text.strip())
            
            # Validate
            try:
                validated = StructuredReview(**result_json)
                return validated.model_dump()
            except:
                return result_json
                
        except json.JSONDecodeError as e:
            return self._create_empty_result(review_text)
        except Exception as e:
            return self._create_empty_result(review_text)
    
    def _create_empty_result(self, review_text: str) -> Dict[str, Any]:
        """Create empty result on failure."""
        return {
            "urlDrugName": "Unknown",
            "condition": "Unknown",
            "rating": 5,
            "effectiveness": "Moderately Effective",
            "sideEffects": "Moderate Side Effects",
            "benefitsReview": None,
            "sideEffectsReview": None,
            "commentsReview": review_text[:500] if review_text else None
        }
    
    def process_dataframe(
        self, 
        df: pd.DataFrame,
        text_column: str = 'combined_review',
        max_reviews: int = None,
        progress_callback=None
    ) -> pd.DataFrame:
        """Process reviews from DataFrame."""
        
        print("\n🤖 Starting AI review structuring...")
        
        if max_reviews:
            df = df.head(max_reviews)
        
        total = len(df)
        all_results = []
        successful = 0
        failed = 0
        
        for i, (idx, row) in enumerate(df.iterrows()):
            # Progress
            print(f"   Processing review {i+1}/{total}...", end=" ")
            
            # Get text
            review_text = row.get(text_column, '')
            if pd.isna(review_text) or not review_text:
                review_text = ' '.join(filter(None, [
                    str(row.get('benefitsReview', '')),
                    str(row.get('sideEffectsReview', '')),
                    str(row.get('commentsReview', ''))
                ]))
            
            context = {
                'drug_name': row.get('urlDrugName'),
                'condition': row.get('condition')
            }
            
            result = self.structure_single(str(review_text), context)
            result['original_index'] = idx
            all_results.append(result)
            
            if result.get('urlDrugName') != 'Unknown':
                successful += 1
                print("✅")
            else:
                failed += 1
                print("⚠️ (using fallback)")
            
            # Rate limiting
            time.sleep(0.5)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        results_df = pd.DataFrame(all_results)
        
        print(f"\n✅ Complete! {successful} successful, {failed} fallback")
        
        return results_df


# ============================================
# ADVERSE EVENT EXTRACTOR (Backward Compatible)
# ============================================

class AdverseEventExtractor(ReviewStructurer):
    """Backward-compatible extractor."""
    
    def extract_from_dataframe(
        self, 
        df: pd.DataFrame,
        batch_size: int = 10,
        max_reviews: int = None,
        progress_callback=None
    ) -> pd.DataFrame:
        """Extract and convert to symptom format."""
        
        structured_df = self.process_dataframe(
            df, 
            max_reviews=max_reviews,
            progress_callback=progress_callback
        )
        
        return self._convert_to_symptom_format(structured_df, df)
    
    def _convert_to_symptom_format(
        self, 
        structured_df: pd.DataFrame,
        original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert to symptom-based rows."""
        rows = []
        
        severity_map = {
            "No Side Effects": None,
            "Mild Side Effects": "mild",
            "Moderate Side Effects": "moderate",
            "Severe Side Effects": "severe",
            "Extremely Severe Side Effects": "life-threatening"
        }
        
        for idx, row in structured_df.iterrows():
            side_effects_category = row.get('sideEffects', 'Moderate Side Effects')
            severity = severity_map.get(side_effects_category)
            
            if severity is None:
                continue
            
            side_effects_text = row.get('sideEffectsReview', '') or ''
            symptoms = self._extract_symptoms_from_text(side_effects_text)
            
            if not symptoms:
                symptoms = [("Reported Side Effects", "Other")]
            
            for symptom_name, body_system in symptoms:
                rows.append({
                    'review_id': row.get('original_index', idx),
                    'drug_name': row.get('urlDrugName', 'Unknown'),
                    'condition_treated': row.get('condition', 'Unknown'),
                    'symptom_name': symptom_name,
                    'severity': severity,
                    'body_system': body_system,
                    'duration': None,
                    'onset': None,
                    'source_quote': side_effects_text[:200] if side_effects_text else '',
                    'overall_sentiment': self._rating_to_sentiment(row.get('rating', 5)),
                    'extraction_confidence': 0.8,
                    'effectiveness': row.get('effectiveness', 'Unknown'),
                    'rating': row.get('rating', 5)
                })
        
        return pd.DataFrame(rows)
    
    def _extract_symptoms_from_text(self, text: str) -> List[tuple]:
        """Extract symptoms from text."""
        if not text:
            return []
        
        text_lower = text.lower()
        symptoms = []
        
        symptom_keywords = {
            'nausea': ('Nausea', 'Gastrointestinal'),
            'vomiting': ('Vomiting', 'Gastrointestinal'),
            'diarrhea': ('Diarrhea', 'Gastrointestinal'),
            'constipation': ('Constipation', 'Gastrointestinal'),
            'stomach': ('Stomach Pain', 'Gastrointestinal'),
            'headache': ('Headache', 'Neurological'),
            'dizziness': ('Dizziness', 'Neurological'),
            'dizzy': ('Dizziness', 'Neurological'),
            'drowsiness': ('Drowsiness', 'Neurological'),
            'drowsy': ('Drowsiness', 'Neurological'),
            'fatigue': ('Fatigue', 'Neurological'),
            'tired': ('Fatigue', 'Neurological'),
            'insomnia': ('Insomnia', 'Psychiatric'),
            'sleep': ('Sleep Disturbance', 'Psychiatric'),
            'anxiety': ('Anxiety', 'Psychiatric'),
            'depression': ('Depression', 'Psychiatric'),
            'mood': ('Mood Changes', 'Psychiatric'),
            'rash': ('Skin Rash', 'Dermatological'),
            'itching': ('Pruritus', 'Dermatological'),
            'weight gain': ('Weight Gain', 'Metabolic'),
            'weight loss': ('Weight Loss', 'Metabolic'),
            'dry mouth': ('Dry Mouth', 'Gastrointestinal'),
            'sweating': ('Excessive Sweating', 'Dermatological'),
            'heart': ('Cardiovascular Effects', 'Cardiovascular'),
            'palpitation': ('Palpitations', 'Cardiovascular'),
            'breathing': ('Respiratory Issues', 'Respiratory'),
            'cough': ('Cough', 'Respiratory'),
            'muscle': ('Muscle Pain', 'Musculoskeletal'),
            'joint': ('Joint Pain', 'Musculoskeletal'),
            'pain': ('Pain', 'Other'),
            'vision': ('Vision Changes', 'Neurological'),
        }
        
        found = set()
        for keyword, (symptom, system) in symptom_keywords.items():
            if keyword in text_lower and symptom not in found:
                symptoms.append((symptom, system))
                found.add(symptom)
        
        return symptoms
    
    def _rating_to_sentiment(self, rating: int) -> str:
        if rating >= 8:
            return 'positive'
        elif rating >= 5:
            return 'neutral'
        else:
            return 'negative'


# ============================================
# MOCK EXTRACTOR
# ============================================

class MockExtractor:
    """Mock extractor for testing without API."""
    
    POSITIVE_KEYWORDS = ['great', 'excellent', 'amazing', 'wonderful', 'perfect', 
                         'life-changing', 'miracle', 'saved', 'love', 'best']
    NEGATIVE_KEYWORDS = ['horrible', 'terrible', 'worst', 'awful', 'hate',
                         'nightmare', 'dangerous', 'never again', 'allergic']
    
    def extract_from_dataframe(self, df: pd.DataFrame, max_reviews: int = None, **kwargs) -> pd.DataFrame:
        print("\n🔧 Running MOCK extraction...")
        
        if max_reviews:
            df = df.head(max_reviews)
        
        rows = []
        
        for idx, row in df.iterrows():
            text = str(row.get('combined_review', '')).lower()
            if not text:
                text = ' '.join([
                    str(row.get('benefitsReview', '')),
                    str(row.get('sideEffectsReview', '')),
                    str(row.get('commentsReview', ''))
                ]).lower()
            
            rating = self._infer_rating(text)
            severity = self._infer_severity(text)
            
            if severity is None:
                continue
            
            symptoms = self._extract_symptoms(text)
            if not symptoms:
                symptoms = [('Reported Side Effects', 'Other')]
            
            for symptom_name, body_system in symptoms:
                rows.append({
                    'review_id': row.get('review_id', idx),
                    'drug_name': row.get('urlDrugName', 'Unknown'),
                    'condition_treated': row.get('condition', 'Unknown'),
                    'symptom_name': symptom_name,
                    'severity': severity,
                    'body_system': body_system,
                    'source_quote': text[:150] + '...',
                    'overall_sentiment': 'positive' if rating >= 7 else 'negative' if rating <= 4 else 'neutral',
                    'extraction_confidence': 0.5,
                    'effectiveness': self._infer_effectiveness(rating),
                    'rating': rating
                })
        
        print(f"✅ Mock extraction: {len(rows)} entries")
        return pd.DataFrame(rows)
    
    def _infer_rating(self, text):
        pos = sum(1 for w in self.POSITIVE_KEYWORDS if w in text)
        neg = sum(1 for w in self.NEGATIVE_KEYWORDS if w in text)
        if pos > neg + 2: return 9
        elif pos > neg: return 7
        elif neg > pos + 2: return 2
        elif neg > pos: return 4
        return 5
    
    def _infer_severity(self, text):
        if any(w in text for w in ['hospital', 'er ', 'emergency', 'suicidal']):
            return 'life-threatening'
        elif 'severe' in text: return 'severe'
        elif any(w in text for w in ['nausea', 'headache', 'dizzy']): return 'moderate'
        elif any(w in text for w in ['dry mouth', 'slight', 'minor']): return 'mild'
        elif 'no side effect' in text: return None
        return 'moderate'
    
    def _infer_effectiveness(self, rating):
        if rating >= 9: return "Highly Effective"
        elif rating >= 7: return "Considerably Effective"
        elif rating >= 5: return "Moderately Effective"
        elif rating >= 3: return "Marginally Effective"
        return "Ineffective"
    
    def _extract_symptoms(self, text):
        symptoms = []
        mapping = {
            'nausea': ('Nausea', 'Gastrointestinal'),
            'headache': ('Headache', 'Neurological'),
            'dizzy': ('Dizziness', 'Neurological'),
            'fatigue': ('Fatigue', 'Neurological'),
            'insomnia': ('Insomnia', 'Psychiatric'),
            'anxiety': ('Anxiety', 'Psychiatric'),
            'rash': ('Skin Rash', 'Dermatological'),
            'dry mouth': ('Dry Mouth', 'Gastrointestinal'),
        }
        found = set()
        for k, (s, sys) in mapping.items():
            if k in text and s not in found:
                symptoms.append((s, sys))
                found.add(s)
        return symptoms


def get_extractor(api_key: str = None, use_mock: bool = False) -> Any:
    """Get appropriate extractor."""
    if use_mock:
        return MockExtractor()
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from config import UF_API_KEY
        if UF_API_KEY or api_key:
            extractor = AdverseEventExtractor(api_key=api_key, timeout=90)
            if extractor.client.initialized:
                return extractor
    except Exception as e:
        print(f"⚠️ Could not initialize API: {e}")
    
    print("⚠️ Falling back to MockExtractor")
    return MockExtractor()


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.data_loader import DrugReviewLoader
    from src.preprocessor import DrugReviewPreprocessor
    
    print("\n" + "🤖 "*20)
    print("PHARMA VIGILANCE - LLM Module Test")
    print("🤖 "*20 + "\n")
    
    loader = DrugReviewLoader()
    train_df, _ = loader.load_all_data()
    
    preprocessor = DrugReviewPreprocessor()
    processed_df = preprocessor.preprocess(train_df)
    
    extractor = get_extractor()
    
    print("\n📊 Testing on 5 reviews...")
    results_df = extractor.extract_from_dataframe(processed_df, max_reviews=5)
    
    print("\n" + "="*60)
    print("📋 RESULTS")
    print("="*60)
    
    if len(results_df) > 0:
        print(f"\nTotal entries: {len(results_df)}")
        print("\n📊 By Body System:")
        print(results_df['body_system'].value_counts().to_string())
        print("\n📊 By Severity:")
        print(results_df['severity'].value_counts().to_string())
