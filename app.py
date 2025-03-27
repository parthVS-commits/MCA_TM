import os
import openai
import pinecone
import streamlit as st
import doublemetaphone
from dotenv import load_dotenv
from difflib import SequenceMatcher
import re
from typing import Dict, List, Tuple
import pycountry
from googletrans import Translator
from better_profanity import profanity
from typing import Dict

# Load environment variables
load_dotenv()

# Initialize API keys
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Dictionary to track namespace information
index_namespaces = {}

try:
    # Create dictionary to map index names to their objects
    index_objects = {}
    
    # Connect to class-objective-all index (uses text-embedding-ada-002)
    index_objects["class-objective-all"] = pc.Index("class-objective-all")
    
    # Connect to tm-test-pipeline1 index (uses text-embedding-3-small)
    index_objects["tm-prod-pipeline"] = pc.Index("tm-prod-pipeline")
    
    # Update this line to use the new index name for MCA validation
    index_objects["mca-scraped-final1"] = pc.Index("mca-scraped-final1")
    
    # Assign the index objects to variables for cleaner code
    class_index = index_objects["class-objective-all"]
    trademark_index = index_objects["tm-prod-pipeline"]
    mca_index = index_objects["mca-scraped-final1"]
            
except Exception as e:
    st.error(f"Failed to initialize Pinecone: {str(e)}")


class TrademarkValidator:
    def __init__(self):
        self.translator = Translator()
        
        # Get all country names
        countries = set(country.name.lower() for country in pycountry.countries)
        
        # Add states/provinces
        states = {
            # Indian States and Union Territories
            'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
            'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka',
            'kerala', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram',
            'nagaland', 'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu',
            'telangana', 'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal',
            'andaman and nicobar', 'chandigarh', 'dadra and nagar haveli',
            'daman and diu', 'delhi', 'jammu and kashmir', 'ladakh', 'lakshadweep',
            'puducherry',
            
            # US States
            'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
            'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
            'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
            'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
            'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
            'new hampshire', 'new jersey', 'new mexico', 'new york',
            'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon',
            'pennsylvania', 'rhode island', 'south carolina', 'south dakota',
            'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
            'west virginia', 'wisconsin', 'wyoming'
        }
        self.articles = {
            'articles': ['the', 'an', 'a','and', 'are', 'our', 'or', 'us', 'we', 'I', 'you', 'me']
               
        } 
        self.restricted_words = {
            'government_words': [
                'board', 'federal', 'municipal', 'forest corporation', 'commission', 
                'republic', 'panchayat', 'development scheme', 'authority', 'president',
                'development authority', 'statute', 'statutory', 'undertaking', 
                'rashtrapati', 'prime minister', 'court', 'judiciary', 'national',
                'small scale industries', 'chief minister', 'governor', 'union',
                'khadi', 'village corporation', 'minister', 'central', 'financial corporation',
                'nation', 'bharat', 'indian'
            ],
            'location_words': countries | states | {
                'asia', 'europe', 'africa', 'australia', 'antarctica', 'north america', 
                'south america'
            }
        }
        
        self.suffixes = {
            'private limited',
            'limited',
            'llp',
            'llc',
            'pvt ltd',
            'pvt. ltd.',
            'p. ltd',
            'pte ltd',
            'ltd'
        }

    def remove_suffix(self, wordmark: str) -> str:
        """Remove common company suffixes and trim whitespace"""
        if not wordmark:
            return None
            
        wordmark_lower = wordmark.lower().strip()
        
        # Check if input is just a suffix
        if wordmark_lower in self.suffixes:
            return None
            
        # Remove suffix if present
        for suffix in self.suffixes:
            if wordmark_lower.endswith(suffix):
                return wordmark[:-(len(suffix))].strip()
                
        return wordmark.strip()

    def check_translations(self, wordmark: str) -> Dict[str, bool]:
        """Check if the wordmark has meaning in Hindi or English"""
        try:
            # Detect language and translate if needed
            en_translation = self.translator.translate(wordmark, dest='en').text
            hi_translation = self.translator.translate(wordmark, dest='hi').text
            
            return {
                'has_meaning': wordmark.lower() != en_translation.lower() or 
                             wordmark != hi_translation,
                'english_meaning': en_translation if wordmark.lower() != en_translation.lower() else None,
                'hindi_meaning': hi_translation if wordmark != hi_translation else None
            }
        except Exception as e:
            return {'error': f"Translation check failed: {str(e)}"}

    def check_location_name(self, wordmark: str) -> Dict[str, bool]:
        """
        Enhanced check for location names that allows locations when part of a larger distinctive name
        """
        if not wordmark:
            return {
                'is_location': False,
                'matched_locations': []
            }
            
        wordmark_lower = wordmark.lower().strip()
        matched_locations = []
        
        # Remove suffix if present to check base name
        base_name = wordmark_lower
        for suffix in self.suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[:-(len(suffix))].strip()
                break
        
        # Split into words
        words = base_name.split()
        
        # If only one word and it's a location, flag it
        if len(words) == 1 and words[0] in self.restricted_words['location_words']:
            matched_locations.append(words[0])
            return {
                'is_location': True,
                'matched_locations': matched_locations
            }
            
        # For multi-word names, check if ALL words are locations
        location_word_count = 0
        for word in words:
            if word in self.restricted_words['location_words']:
                location_word_count += 1
                matched_locations.append(word)
        
        # If all words are locations (excluding common joining words), flag it
        non_joining_words = [w for w in words if w not in {'of', 'and', 'the', '&'}]
        if location_word_count == len(non_joining_words) and location_word_count > 0:
            return {
                'is_location': True,
                'matched_locations': matched_locations
            }
        
        # If there are additional distinctive words along with location(s), it's okay
        return {
            'is_location': False,
            'matched_locations': matched_locations
        }

    def check_government_patronage(self, wordmark: str) -> Dict[str, bool]:
        """Check for government patronage implications"""
        words = wordmark.lower().split()
        restricted_matches = []
        
        for word in words:
            if word in self.restricted_words['government_words']:
                restricted_matches.append(word)
        
        return {
            'implies_patronage': len(restricted_matches) > 0,
            'restricted_words_found': restricted_matches
        }
    
    def check_articles(self, wordmark: str) -> Dict[str, bool]:
        """
        Check if the wordmark ONLY contains articles/pronouns
        Only flag as invalid if the entire wordmark is just articles/pronouns
        """
        if not wordmark:
            return {
                'has_articles': False,
                'restricted_words_found': []
            }
            
        words = wordmark.lower().split()
        
        # If the wordmark has multiple words, check if ALL words are articles/pronouns
        if len(words) > 0:
            restricted_matches = []
            
            # Check each word
            for word in words:
                if word in self.articles['articles']:
                    restricted_matches.append(word)
            
            # Only flag as invalid if ALL words are articles/pronouns
            all_words_are_articles = len(restricted_matches) == len(words)
            
            return {
                'has_articles': all_words_are_articles,
                'restricted_words_found': restricted_matches if all_words_are_articles else []
            }
        
        # Single word case - check if it's in the articles list
        return {
            'has_articles': wordmark.lower() in self.articles['articles'],
            'restricted_words_found': [wordmark.lower()] if wordmark.lower() in self.articles['articles'] else []
        }

    def check_similar_existing_companies(self, wordmark: str, company_names: List[str]) -> Dict[str, List[str]]:
        """Check for similarity with existing company names"""
        similar_names = []
        for name in company_names:
            # Check for exact match
            if wordmark.lower() == name.lower():
                similar_names.append(('exact_match', name))
                continue
            
            # Check for place name difference
            base_name = re.sub(r'\b[A-Z][a-z]+ (Private Limited|Limited|LLP|LLC)\b', '', name)
            if base_name.lower() == wordmark.lower():
                similar_names.append(('place_difference', name))
        
        return {
            'has_similarities': len(similar_names) > 0,
            'similar_names': similar_names
        }

    def check_embassy_connections(self, wordmark: str) -> Dict[str, bool]:
        """Check for connections with foreign embassies/consulates"""
        embassy_related_terms = {
            'embassy', 'diplomatic', 'ambassador', 'diplomatic mission'
        }
        
        words = wordmark.lower().split()
        matches = []
        
        for term in embassy_related_terms:
            if term in ' '.join(words):
                matches.append(term)
                
        return {
            'has_embassy_connection': len(matches) > 0,
            'matched_terms': matches
        }

    def validate_trademark(self, wordmark: str, existing_companies: List[str] = None) -> Dict[str, Dict]:
        """Perform comprehensive trademark validation"""
        if existing_companies is None:
            existing_companies = []
            
        results = {
            'translation_check': self.check_translations(wordmark),
            'location_check': self.check_location_name(wordmark),
            'government_check': self.check_government_patronage(wordmark),
            'company_similarity': self.check_similar_existing_companies(wordmark, existing_companies),
            'embassy_check': self.check_embassy_connections(wordmark),
            'articles_check': self.check_articles(wordmark)
        }
        
        # Determine overall validity
        is_valid = all([
            not results['location_check']['is_location'],
            not results['government_check']['implies_patronage'],
            not results['company_similarity']['has_similarities'],
            not results['embassy_check']['has_embassy_connection'],
            not results['articles_check']['has_articles']
        ])
        
        results['overall_validity'] = {
            'is_valid': is_valid,
            'validation_messages': self._generate_validation_messages(results)
        }
        
        return results

    def _generate_validation_messages(self, results: Dict) -> List[str]:
        """Generate human-readable validation messages"""
        messages = []
        
        if results['translation_check'].get('has_meaning'):
            messages.append(f"Warning: Wordmark has meaning in other languages")
            
        if results['location_check']['is_location']:
            messages.append(f"Invalid: Contains location name(s): {', '.join(results['location_check']['matched_locations'])}")
            
        if results['government_check']['implies_patronage']:
            messages.append(f"Invalid: Uses restricted government terms: {', '.join(results['government_check']['restricted_words_found'])}")
            
        if results['company_similarity']['has_similarities']:
            messages.append("Invalid: Similar to existing company names")
            
        if results['embassy_check']['has_embassy_connection']:
            messages.append(f"Invalid: Suggests embassy/consulate connection: {', '.join(results['embassy_check']['matched_terms'])}")

        if results['articles_check']['has_articles']:
            messages.append(f"Invalid: Articles/Pronouns Found")
            
        return messages

def validate_suggestion(suggestion: str, validator: TrademarkValidator, similarity_threshold: float = 0.6) -> Dict[str, bool]:
    """
    Comprehensively validate a single suggestion with stricter similarity checks
    """
    try:
        # Clean the suggestion
        cleaned_suggestion = validator.remove_suffix(suggestion)
        if not cleaned_suggestion:
            return {"is_valid": False, "reason": "Invalid name format"}

        # Check for profanity
        if profanity.contains_profanity(cleaned_suggestion):
            return {"is_valid": False, "reason": "Contains inappropriate content"}

        # Run all validation checks
        validation_results = validator.validate_trademark(cleaned_suggestion)
        
        # Enhanced similarity check with MCA database
        matches = check_multiple_phonetic_matches(cleaned_suggestion, mca_index)
        
        if matches:
            # Check for exact matches
            for match in matches:
                if match["Matching Wordmark"].lower() == cleaned_suggestion.lower():
                    return {"is_valid": False, "reason": "Exact match found in database"}
                
            # Check for high phonetic similarity
            for match in matches:
                if match["Phonetic Score"] > 0.85:
                    return {"is_valid": False, "reason": "High phonetic similarity with existing name"}
                
            # Check for high semantic similarity
            for match in matches:
                if match["Semantic Score"] > similarity_threshold:
                    return {"is_valid": False, "reason": "High semantic similarity with existing name"}
                
            # Check for high hybrid similarity
            for match in matches:
                if match["Hybrid Score"] > similarity_threshold:
                    return {"is_valid": False, "reason": "High overall similarity with existing name"}

        # Additional validation criteria
        is_valid = (
            validation_results['overall_validity']['is_valid'] and
            len(cleaned_suggestion) >= 3 and
            not any(char.isdigit() for char in cleaned_suggestion)
        )

        return {
            "is_valid": is_valid,
            "reason": None if is_valid else "Failed validation checks",
            "validation_details": validation_results
        }

    except Exception as e:
        return {"is_valid": False, "reason": f"Validation error: {str(e)}"}

def get_phonetic_representation(word):
    word_lowercase = word.lower()
    primary, secondary = doublemetaphone.doublemetaphone(word_lowercase)
    return primary or secondary or word_lowercase

def get_embedding(text, model="text-embedding-3-small"):
    try:
        if not text or text.isspace():
            st.error("Please enter a valid company name that is not just a suffix (like 'private limited', 'ltd', etc.)")
            return None
        
        normalized_text = text.lower()
            
        response = openai.Embedding.create(
            model=model,
            input=[normalized_text]
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        # Try fallback to older model if the new one fails
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=[normalized_text]
            )
            return response["data"][0]["embedding"]
        except Exception as e2:
            st.error(f"Fallback embedding also failed: {str(e2)}")
            return None

def calculate_phonetic_similarity(word1, word2):
    phonetic1 = get_phonetic_representation(word1)
    phonetic2 = get_phonetic_representation(word2)
    return SequenceMatcher(None, phonetic1, phonetic2).ratio()

def calculate_hybrid_score(phonetic_score, semantic_score, phonetic_weight=0.6, semantic_weight=0.4):
    return (phonetic_weight * phonetic_score) + (semantic_weight * semantic_score)

def check_multiple_phonetic_matches(wordmark, index, model="text-embedding-3-small", namespace=""):
    try:
        # Initialize validator
        validator = TrademarkValidator()
        
        # Check if input is just a suffix
        cleaned_wordmark = validator.remove_suffix(wordmark)
        if not cleaned_wordmark:
            st.error("Please enter a valid company name that is not just a suffix (like 'private limited', 'ltd', etc.)")
            return None
              
        cleaned_wordmark = cleaned_wordmark.lower()
     
        phonetic_representation = get_phonetic_representation(cleaned_wordmark)
        input_embedding = get_embedding(cleaned_wordmark, model=model)

        if input_embedding is None:
            return None

        query_result = index.query(
            vector=input_embedding,
            top_k=5,
            include_metadata=True,
            namespace=namespace
        )

        matches = []
        for match in query_result["matches"]:
            # Handle different metadata formats between MCA and Trademark indexes
            if 'original_data' in match.get('metadata', {}):
                # MCA index format
                stored_wordmark = match["metadata"].get("original_data", "")
                stored_phonetic = match["metadata"].get("phonetic_representation", "")
                stored_classes = None  # MCA doesn't have class info
            else:
                # Trademark index format
                stored_wordmark = match.get('metadata', {}).get('wordMark', '')
                stored_phonetic = match.get('metadata', {}).get('Phonetic_Representation', '')
                stored_classes = match.get('metadata', {}).get('wclass', [])

            # Calculate similarity using cleaned names
            phonetic_score = calculate_phonetic_similarity(cleaned_wordmark, stored_wordmark.lower())
            semantic_score = match["score"]
            hybrid_score = calculate_hybrid_score(phonetic_score, semantic_score)

            matches.append({
                "Matching Wordmark": stored_wordmark,
                "Cleaned Wordmark": validator.remove_suffix(stored_wordmark),
                "Phonetic Representation": stored_phonetic,
                "Class": stored_classes,
                "Phonetic Score": phonetic_score,
                "Semantic Score": semantic_score,
                "Hybrid Score": hybrid_score
            })

        matches = sorted(matches, key=lambda x: x["Hybrid Score"], reverse=True)
        return matches
    except Exception as e:
        st.error(f"Error checking phonetic matches: {str(e)}")
        return None
    
def suggest_similar_names(wordmark):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a creative naming assistant who generates highly unique and distinctive business names. 
                    Follow these strict guidelines:
                    1. Generate exactly 5 unique name suggestions
                    2. Each suggestion must be SIGNIFICANTLY different from the input name
                    3. Avoid names that sound similar to the input
                    4. Create completely new and unique word combinations
                    5. Don't use common prefixes or suffixes of the input name
                    6. Avoid location names, government terms, or restricted words
                    7. Keep names between 3-20 characters
                    8. Don't use numbers or special characters
                    9. Ensure names are pronounceable and memorable
                    10. Don't include common suffixes (Ltd, Private Limited, etc.)
                    11. Each suggestion should have a distinct sound and appearance
                    12. Avoid using parts or variations of the input name
                    
                    Focus on creating entirely new, distinctive names rather than variations of the existing name."""
                },
                {
                    "role": "user",
                    "content": f"Create five completely unique and distinctive business names that are very different from '{wordmark}'. The names should be suitable for business registration and have no similarity to existing company names."
                }
            ],
            max_tokens=150,
            n=1,
            temperature=0.9  # Increased for more creative/diverse results
        )
        suggestions = response.choices[0].message.content.strip().split("\n")
        return [name.strip() for name in suggestions if name.strip()]
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")
        return []

def validate_suggestions(suggestions, indexes, similarity_threshold=0.9):
    """
    Validate suggestions to ensure they are unique.
    """
    try:
        unique_suggestions = []
        for suggestion in suggestions:
            is_unique = True
            input_embedding = get_embedding(suggestion, model="text-embedding-3-small")
            if input_embedding:
                for index in indexes:
                    query_result = index.query(
                        vector=input_embedding,
                        top_k=1,
                        include_metadata=True
                    )

                    if query_result.get('matches'):
                        # If the match score is above the threshold, we consider it a duplicate
                        highest_match = query_result['matches'][0]
                        if highest_match.get('score', 0) >= similarity_threshold:
                            is_unique = False
                            break
            if is_unique:
                unique_suggestions.append(suggestion)
        return unique_suggestions
    except Exception as e:
        st.error(f"Error validating suggestions: {str(e)}")
        return []

def suggest_similar_names(wordmark):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative naming assistant who generates unique and meaningful alternative names for businesses. Your goal is to provide exactly 3 unique name suggestions by modifying the input name using a prefix, suffix, or an additional word while ensuring that the essence of the original name remains intact and also provide other two names solely based on the context."
                                "Understand the context of the input name before suggesting alternatives."
                                "Prioritize context over language, but if the input has a strong linguistic influence, consider that in your suggestions (not mandatory)."
                                "If using synonyms, ensure they match the language of the input."
                                "Do not override any instruction—integrate all requirements naturally."
                                "Your focus: Creativity, uniqueness, and relevance to the original name."
                },
                {
                    "role": "user",
                    "content": f"Suggest five creative and unique alternative names for the word '{wordmark}'."
                }
            ],
            max_tokens=50,
            n=1
        )
        suggestions = response.choices[0].message.content.strip().split("\n")
        return [name.strip() for name in suggestions if name.strip()]
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")
        return []

def validate_suggestions(suggestions, similarity_threshold=0.9):
    try:
        unique_suggestions = []
        for suggestion in suggestions:
            is_unique = True
            input_embedding = get_embedding(suggestion)
            if input_embedding:
                query_result = trademark_index.query(
                    vector=input_embedding,
                    top_k=1,
                )

                if query_result.matches and query_result.matches[0].score >= similarity_threshold:
                    is_unique = False

            if is_unique:
                unique_suggestions.append(suggestion)
        return unique_suggestions
    except Exception as e:
        st.error(f"Error validating suggestions: {str(e)}")
        return []

def get_unique_suggestions(input_wordMark, max_retries=5):
    suggestions = suggest_similar_names(input_wordMark)
    unique_suggestions = validate_suggestions(suggestions)

    retries = 0
    while len(unique_suggestions) < 5 and retries < max_retries:
        new_suggestions = suggest_similar_names(input_wordMark)
        unique_suggestions += validate_suggestions(new_suggestions)
        unique_suggestions = list(set(unique_suggestions))
        retries += 1

    return unique_suggestions[:5]

def extract_class_number(trademark_class_text: str) -> str:
    """
    Extract the class number from GPT-4's classification response
    """
    try:
        import re
        # First look for patterns like "Class 41" or "Class: 41"
        class_match = re.search(r'Class\s*:?\s*(\d+)', trademark_class_text, re.IGNORECASE)
        if class_match:
            return class_match.group(1)
        
        # Then look for just numbers at the start of the text
        number_match = re.search(r'^\d+', trademark_class_text)
        if number_match:
            return number_match.group(0)
        
        # Finally look for any numbers in the text
        any_number = re.search(r'\d+', trademark_class_text)
        if any_number:
            return any_number.group(0)
            
        return None
    except Exception as e:
        st.error(f"Error extracting class number: {str(e)}")
        return None

def classify_objective_gpt4(objective: str):
    """
    Use GPT-4 to classify the business objective
    """
    if not objective or len(objective) < 5:
        return "Invalid input. Please provide a meaningful objective."

    # Use the appropriate embedding model to match the class-objective-all index 
    response = openai.Embedding.create(
        input=objective.strip().lower(),
        model="text-embedding-ada-002"  # This index uses ada-002
    )
    query_embedding = response["data"][0]["embedding"]

    # Simplified query without namespace parameter
    results = class_index.query(
        vector=query_embedding, 
        top_k=5, 
        include_metadata=True
    )

    if not results.get("matches"):
        return "No suitable class found."

    matched_classes = [
        f"Class {m['id']}: {m['metadata']['description']}"
        for m in results["matches"]
    ]
    context = "\n".join(matched_classes)

    prompt = f"""
    You are an expert in trademark classification. Given the following objective:

    "{objective}"

    And these possible trademark classes:
    {context}
    """

    gpt_response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are a trademark classification assistant."},
                  {"role": "user", "content": prompt}]
    )

    return gpt_response["choices"][0]["message"]["content"]

def check_class_specific_matches(wordmark: str, trademark_class: str):
    """
    Check for similar marks specifically within the identified trademark class
    """
    try:
        # Use the correct embedding model for this index
        input_embedding = get_embedding(wordmark, model="text-embedding-3-small")
        if input_embedding is None:
            return None
            
        # Extract just the class number if full classification text is provided
        class_number = extract_class_number(trademark_class)
        if not class_number:
            st.error("Could not extract class number from classification")
            return None
            
        # Query without namespace parameter
        query_result = trademark_index.query(
            vector=input_embedding,
            top_k=50,
            include_metadata=True
        )
        
        # Filter the results manually
        filtered_matches = []
        for match in query_result.get('matches', []):
            stored_classes = match.get('metadata', {}).get('wclass', [])
            
            # Check if this result matches our class
            is_match = False
            if isinstance(stored_classes, list):
                if class_number in stored_classes:
                    is_match = True
            elif stored_classes == class_number:  # Handle case of single string
                is_match = True
                
            if is_match:
                filtered_matches.append(match)
                
        # Process the filtered matches
        matches = []
        for match in filtered_matches:
            metadata = match.get('metadata', {})
            stored_wordmark = metadata.get('wordMark', '')
            stored_classes = metadata.get('wclass', [])
            
            phonetic_score = calculate_phonetic_similarity(wordmark, stored_wordmark)
            semantic_score = match.get('score', 0)
            hybrid_score = calculate_hybrid_score(phonetic_score, semantic_score)
            
            matches.append({
                "Matching Wordmark": stored_wordmark,
                "Class": stored_classes,
                "Phonetic Score": phonetic_score,
                "Semantic Score": semantic_score,
                "Hybrid Score": hybrid_score,
                "Is Same Class": True  # We manually filtered these
            })
            
        return sorted(matches, key=lambda x: x["Hybrid Score"], reverse=True)
            
    except Exception as e:
        st.error(f"Error in class-specific search: {str(e)}")
        return None

def classify_and_check_trademark(objective: str, wordmark: str, trademark_class: str):
    """
    Comprehensive trademark validation that performs multiple checks after classification
    """
    try:
        # First check: Search within the specific trademark class
        class_specific_matches = check_class_specific_matches(wordmark, trademark_class)
        
        # Second check: Search across all classes for similar marks
        cross_class_matches = check_multiple_phonetic_matches(wordmark, trademark_index, model="text-embedding-3-small")
        
        results = {
            'class_matches': [],
            'cross_class_matches': [],
            'high_risk_matches': [],
            'suggestions_needed': False
        }
        
        # Process class-specific matches
        if class_specific_matches:
            results['class_matches'] = [
                match for match in class_specific_matches 
                if match["Hybrid Score"] > 0.6
            ]
            
        # Process cross-class matches
        if cross_class_matches:
            results['cross_class_matches'] = [
                match for match in cross_class_matches 
                if match["Hybrid Score"] > 0.7
            ]
            
        # Identify high-risk matches
        high_risk_matches = [
            match for match in (results['class_matches'] + results['cross_class_matches'])
            if match["Hybrid Score"] > 0.8
        ]
        
        results['high_risk_matches'] = high_risk_matches
        results['suggestions_needed'] = bool(high_risk_matches)
        
        return results
        
    except Exception as e:
        st.error(f"Error in trademark validation: {str(e)}")
        return None
    

def main():
    st.title("Integrated MCA and Trademark Validation")
    st.write("This tool validates your name against MCA and Trademark databases.")

    validator = TrademarkValidator()
    
    col1, col2 = st.columns(2)
    with col1:
        wordmark = st.text_input("Enter the Name/Wordmark:", "")
    with col2:
        objective = st.text_input("Enter the Business Objective:", "")

    if st.button("Validate"):
        if not wordmark:
            st.warning("Please enter the Name/Wordmark.")
            return
            
        # First check if input is just a suffix
        cleaned_name = validator.remove_suffix(wordmark)
        if not cleaned_name:
            st.error("Please enter a valid name. The input cannot be just a suffix (like 'private limited', 'ltd', etc.)")
            return

        # Create tabs for MCA and Trademark validation results
        mca_tab, trademark_tab, suggestions_tab = st.tabs(["MCA Validation", "Trademark Validation", "Suggestions"])
        
        with mca_tab:
            st.header("MCA Name Validation")
            
            # Perform MCA name validation
            with st.spinner("Performing MCA name validation..."):
                validation_results = validator.validate_trademark(wordmark)
                
                # Get the correct namespace for MCA
                try:
                    index_stats = mca_index.describe_index_stats()
                    namespaces = list(index_stats.get('namespaces', {}).keys())
                    namespace_to_use = namespaces[0] if namespaces else ""
                except Exception as e:
                    st.error(f"Failed to connect to Pinecone MCA index: {str(e)}")
                    namespace_to_use = ""
                    
                # Check for similar names in MCA database
                mca_matches = check_multiple_phonetic_matches(wordmark, mca_index, namespace=namespace_to_use)
            
            # Display MCA validation results
            if not validation_results['overall_validity']['is_valid']:
                st.error("### ⚠️ MCA Validation Issues Found!")
                for message in validation_results['overall_validity']['validation_messages']:
                    st.warning(message)
                        
                if validation_results['translation_check'].get('has_meaning'):
                    with st.expander("Translation Details"):
                        if validation_results['translation_check'].get('english_meaning'):
                            st.write(f"English meaning: {validation_results['translation_check']['english_meaning']}")
                        if validation_results['translation_check'].get('hindi_meaning'):
                            st.write(f"Hindi meaning: {validation_results['translation_check']['hindi_meaning']}")
            
            # Display MCA database match results
            if mca_matches:
                # Filter high-risk matches based on hybrid score threshold
                high_risk_matches = [match for match in mca_matches if match["Hybrid Score"] > 0.8]

                if high_risk_matches:
                    st.error("### ⚠️ High Risk MCA Name Matches Found!")
                    for match in high_risk_matches:
                        with st.expander(f"Match: {match['Matching Wordmark']}"):
                            st.write(f"- **Phonetic Representation:** {match['Phonetic Representation']}")
                            st.write(f"- **Phonetic Score:** {match['Phonetic Score']:.2f}")
                            st.write(f"- **Semantic Score:** {match['Semantic Score']:.2f}")
                            st.write(f"- **Hybrid Score:** {match['Hybrid Score']:.2f}")
                else:
                    moderate_risk_matches = [match for match in mca_matches if match["Hybrid Score"] > 0.75]
                    if moderate_risk_matches:
                        st.warning("### ⚠️ Similar MCA Names Found (Moderate Risk)")
                        for match in moderate_risk_matches:
                            with st.expander(f"Similar Name: {match['Matching Wordmark']}"):
                                st.write(f"- **Phonetic Representation:** {match['Phonetic Representation']}")
                                st.write(f"- **Phonetic Score:** {match['Phonetic Score']:.2f}")
                                st.write(f"- **Semantic Score:** {match['Semantic Score']:.2f}")
                                st.write(f"- **Hybrid Score:** {match['Hybrid Score']:.2f}")
                    else:
                        st.success("✅ No high-risk MCA name matches found!")
            else:
                if validation_results['overall_validity']['is_valid']:
                    st.success("✅ MCA Validation Passed! No issues found with the name.")
                else:
                    st.info("No similar MCA names found in database, but other validation issues exist.")
                
            # Display detailed MCA validation report
            with st.expander("View Detailed MCA Validation Report"):
                st.write("### Validation Checks Performed:")
                
                st.write("1. **Translation Check:**")
                if validation_results['translation_check'].get('has_meaning'):
                    st.write("- Has meaning in other languages")
                    if validation_results['translation_check'].get('english_meaning'):
                        st.write(f"- English: {validation_results['translation_check']['english_meaning']}")
                    if validation_results['translation_check'].get('hindi_meaning'):
                        st.write(f"- Hindi: {validation_results['translation_check']['hindi_meaning']}")
                else:
                    st.write("- No significant translations found")
                
                st.write("\n2. **Location Name Check:**")
                if validation_results['location_check']['is_location']:
                    st.write(f"- Contains location names: {', '.join(validation_results['location_check']['matched_locations'])}")
                else:
                    st.write("- No location names found")
                
                st.write("\n3. **Government Terms Check:**")
                if validation_results['government_check']['implies_patronage']:
                    st.write(f"- Contains restricted terms: {', '.join(validation_results['government_check']['restricted_words_found'])}")
                else:
                    st.write("- No restricted government terms found")
                
                st.write("\n4. **Embassy/Consulate Connection Check:**")
                if validation_results['embassy_check']['has_embassy_connection']:
                    st.write(f"- Contains embassy-related terms: {', '.join(validation_results['embassy_check']['matched_terms'])}")
                else:
                    st.write("- No embassy/consulate connections found")

                st.write("\n5. **Articles/Pronouns Check:**")
                if validation_results['articles_check']['has_articles']:
                    st.write(f"- Contains articles/pronouns")
                else:
                    st.write("- No articles/pronouns found")
                
        with trademark_tab:
            st.header("Trademark Validation")
            
            if not objective:
                st.warning("Please enter a business objective for trademark validation.")
            else:
                # Step 1: Classify the objective
                with st.spinner("Classifying business objective..."):
                    trademark_class_text = classify_objective_gpt4(objective)
                    class_number = extract_class_number(trademark_class_text)
                    
                    if not class_number:
                        st.error("Could not determine trademark class number.")
                    else:
                        # Display the classification result
                        st.success("### Classification Result")
                        st.write(f"**Objective Analysis:** {trademark_class_text}")
                        st.info(f"**Mapped Trademark Class:** {class_number}")

                        # Step 2: Check for similar trademarks
                        with st.spinner("Checking trademark database..."):
                            trademark_results = classify_and_check_trademark(objective, wordmark, class_number)

                        if trademark_results:
                            # Display results based on risk level
                            if trademark_results['high_risk_matches']:
                                # Only show high-risk matches if they exist
                                st.error("### ⚠️ High Risk Trademark Matches Found!")
                                for match in trademark_results['high_risk_matches']:
                                    with st.expander(f"High Risk Match: {match['Matching Wordmark']}"):
                                        st.write(f"- **Class:** {match.get('Class', 'N/A')}")
                                        st.write(f"- **Phonetic Score:** {match['Phonetic Score']:.2f}")
                                        st.write(f"- **Semantic Score:** {match['Semantic Score']:.2f}")
                                        st.write(f"- **Hybrid Score:** {match['Hybrid Score']:.2f}")
                            elif trademark_results['class_matches']:
                                # Only show class-specific matches if no high-risk matches exist
                                st.warning("### Similar Trademarks Found in Same Class")
                                for match in trademark_results['class_matches']:
                                    with st.expander(f"Match: {match['Matching Wordmark']}"):
                                        st.write(f"- **Class:** {match['Class']}")
                                        st.write(f"- **Phonetic Score:** {match['Phonetic Score']:.2f}")
                                        st.write(f"- **Semantic Score:** {match['Semantic Score']:.2f}")
                                        st.write(f"- **Hybrid Score:** {match['Hybrid Score']:.2f}")
                            else:
                                st.success("✅ No high-risk trademark matches found!")
                        else:
                            st.info("No trademark validation results available.")

        with suggestions_tab:
            st.header("Alternative Name Suggestions")
            
            # Determine if we need to generate suggestions
            need_suggestions = False
            
            # Check MCA validation results
            if (not validation_results['overall_validity']['is_valid'] or 
                (mca_matches and any(match["Hybrid Score"] > 0.8 for match in mca_matches))):
                need_suggestions = True
            
            # Check trademark validation results (if objective was provided)
            if objective and class_number:
                if trademark_results and trademark_results['suggestions_needed']:
                    need_suggestions = True
            
            if need_suggestions:
                with st.spinner("Generating alternative name suggestions..."):
                    # Generate suggestions considering both MCA and trademark validation
                    unique_suggestions = get_unique_suggestions(wordmark)

                    if unique_suggestions:
                        st.success("### ✅ Alternative Name Suggestions:")
                        for suggestion in unique_suggestions:
                            st.write(f"- {suggestion}")
                    else:
                        st.warning("No unique alternative suggestions could be generated.")
            else:
                st.success("✅ No suggestions needed - current name appears to be valid!")

if __name__ == "__main__":
    main()
