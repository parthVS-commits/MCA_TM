import os
import openai
from pinecone import Pinecone
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
# from elasticapm.contrib.starlette import make_apm_client, ElasticAPM
import hashlib
import requests
import time
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# @st.cache_resource
# def initialize_apm():
#     apm_config = {
#         'SERVICE_NAME': os.environ.get('ES_SERVICE_NAME_FE'),
#         'SECRET_TOKEN': os.environ.get('ES_SECRET_TOKEN'),
#         'SERVER_URL': os.environ.get('ES_SERVER_URL'),
#         'ENVIRONMENT': 'prod',
#     }
#     return Client(apm_config)

# try:
#     apm = initialize_apm() 
# except:
#     apm = None  # Handle cases where APM is not available

# Initialize API keys
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Dictionary to track namespace information
index_namespaces = {}

# Simple in-memory cache for web search results
fallback_cache = {}
CACHE_DURATION = timedelta(hours=24)

try:
    # Create dictionary to map index names to their objects
    index_objects = {}
    
    # Connect to class-objective-all index (uses text-embedding-ada-002)
    index_objects["class-objective-all"] = pc.Index("class-objective-all")
    
    # Connect to tm-prod-pipeline index (uses text-embedding-3-small)
    index_objects["tm-prod-pipeline"] = pc.Index("tm-prod-pipeline")
    
    # Connect to MCA index
    index_objects["mca-scraped-final1"] = pc.Index("mca-scraped-final1")
    
    # Assign the index objects to variables for cleaner code
    class_index = index_objects["class-objective-all"]
    trademark_index = index_objects["tm-prod-pipeline"]
    mca_index = index_objects["mca-scraped-final1"]
            
except Exception as e:
    st.error(f"Failed to initialize Pinecone: {str(e)}")


# ===============================
# BRAVE API WEB SEARCH INTEGRATION
# ===============================

class BraveZaubaCorpSearcher:
    """Enhanced Brave API ZaubaCorp search with exact matching"""
    
    def __init__(self):
        self.brave_api_key = os.environ.get("BRAVE_API_KEY")
        if not self.brave_api_key:
            st.warning("BRAVE_API_KEY not configured - web search will be limited")
        else:
            st.info("🌐 Brave API ZaubaCorp searcher initialized")

    def normalize_company_name(self, company_name: str) -> str:
        """Enhanced normalization for company name matching"""
        if not company_name:
            return ""
        
        normalized = company_name.lower().strip()
        
        # Remove common suffixes for comparison
        suffixes_to_remove = [
            'private limited', 'pvt ltd', 'pvt. ltd.', 'pvt ltd.',
            'limited', 'ltd', 'llp', 'corporation', 'corp',
            'company', 'co', 'inc', 'private'
        ]
        
        for suffix in suffixes_to_remove:
            if normalized.endswith(' ' + suffix):
                normalized = normalized[:-len(' ' + suffix)].strip()
                break
            elif normalized == suffix:
                return ""
            elif normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
                break
        
        # Clean up extra spaces and remove special characters
        normalized = ' '.join(normalized.split())
        normalized = normalized.replace('(', '').replace(')', '')
        normalized = normalized.replace('[', '').replace(']', '')
        normalized = ' '.join(normalized.split())
        
        return normalized

    def extract_company_name_from_title(self, title: str) -> str:
        """Extract company name from ZaubaCorp title with comprehensive pattern matching"""
        if not title:
            return ""
        
        title = title.strip()
        
        # ZaubaCorp title patterns
        patterns = [
            " - Company, directors and contact details",
            " - Company, directors and co",
            " - Company, directors",
            " | Investors, Shareholders, Directors and Contact Details",
            " | Investors, Shareholders, Directors",
            " | Investors, Shareholders",
            " | Company Info",
            " | ZaubaCorp",
            " | Zauba Corp",
            " - Directors and Contact Details",
            " - Directors",
            " - Contact Details",
            " | Directors of ",
            " | ",
            " - ",
            "|",
            " –",
            " —"
        ]
        
        # Try each pattern
        for pattern in patterns:
            if pattern in title:
                if "Directors of " in title and title.startswith("Directors of "):
                    company_name = title.replace("Directors of ", "").strip()
                    return company_name
                else:
                    company_name = title.split(pattern)[0].strip()
                    return company_name
        
        return title

    def is_exact_match(self, target: str, candidate: str) -> bool:
        """Enhanced exact match detection"""
        if not target or not candidate:
            return False
        
        if target == candidate:
            return True
        
        # Word set comparison
        target_words = set(target.split())
        candidate_words = set(candidate.split())
        
        if len(target_words) == len(candidate_words) and target_words == candidate_words:
            return True
        
        # Ignore common articles
        articles = {'the', 'a', 'an', 'and', '&', 'of', 'for', 'with', 'in', 'on', 'at'}
        target_significant = target_words - articles
        candidate_significant = candidate_words - articles
        
        if target_significant and candidate_significant:
            if len(target_significant) == len(candidate_significant) and target_significant == candidate_significant:
                return True
        
        return False

    def search_zaubacorp_brave(self, company_name: str) -> List[Dict]:
        """Brave API search with targeted queries for exact matching"""
        
        if not self.brave_api_key:
            return []
        
        search_queries = [
            f'"{company_name}" site:zaubacorp.com',
            f'"{company_name} Private Limited" site:zaubacorp.com',
            f'"{company_name} Pvt Ltd" site:zaubacorp.com'
        ]
        
        all_results = []
        
        for query in search_queries:
            try:
                url = "https://api.search.brave.com/res/v1/web/search"
                headers = {
                    "Accept": "application/json",
                    "X-Subscription-Token": self.brave_api_key
                }
                
                params = {
                    "q": query,
                    "count": 10,
                    "country": "IN",
                    "search_lang": "en"
                }
                
                response = requests.get(url, headers=headers, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    for result in data.get("web", {}).get("results", []):
                        if "zaubacorp.com" in result.get("url", "").lower():
                            all_results.append({
                                "title": result.get("title", ""),
                                "snippet": result.get("description", ""),
                                "url": result.get("url", ""),
                                "query": query,
                                "source": "brave_zaubacorp"
                            })
                elif response.status_code == 429:
                    time.sleep(1)
                    continue
                    
                time.sleep(0.3)
                
            except Exception as e:
                continue
        
        return all_results

    def parse_zaubacorp_results(self, company_name: str, search_results: List[Dict]) -> Dict[str, any]:
        """Enhanced parsing with fixed name extraction"""
        
        if not search_results:
            return {
                "found_on_zaubacorp": False,
                "evidence_type": "no_results",
                "evidence_details": "No ZaubaCorp results found",
                "confidence": 0.0
            }
        
        target_company = self.normalize_company_name(company_name)
        exact_matches = []
        confidence_score = 0.0
        
        for result in search_results:
            title = result.get("title", "")
            url = result.get("url", "")
            
            extracted_name = self.extract_company_name_from_title(title)
            normalized_extracted = self.normalize_company_name(extracted_name)
            
            if self.is_exact_match(target_company, normalized_extracted):
                exact_matches.append({
                    "company_name": extracted_name,
                    "source_title": title,
                    "match_type": "exact",
                    "url": url
                })
                confidence_score += 1.0
        
        found_on_zaubacorp = len(exact_matches) > 0
        
        if found_on_zaubacorp:
            evidence_details = f"Found {len(exact_matches)} exact matches"
            evidence_type = "exact_match"
        else:
            evidence_details = f"No exact matches found among {len(search_results)} results"
            evidence_type = "no_exact_match"
        
        return {
            "found_on_zaubacorp": found_on_zaubacorp,
            "evidence_type": evidence_type,
            "evidence_details": evidence_details,
            "confidence": min(confidence_score, 1.0),
            "total_results": len(search_results),
            "exact_matches": exact_matches
        }

    # @capture_span()
    def comprehensive_zaubacorp_check(self, company_name: str) -> Dict[str, any]:
        """Enhanced comprehensive check with fixed extraction"""
        
        try:
            search_results = self.search_zaubacorp_brave(company_name)
            analysis = self.parse_zaubacorp_results(company_name, search_results)
            
            return {
                "has_conflicts": analysis["found_on_zaubacorp"],
                "risk_level": "HIGH" if analysis["found_on_zaubacorp"] else "LOW",
                "evidence_summary": analysis["evidence_details"],
                "method": "brave_zaubacorp_enhanced_extraction",
                "web_search_performed": True,
                "search_method": "brave_api_zaubacorp_enhanced",
                "zaubacorp_analysis": analysis
            }
            
        except Exception as e:
            return {
                "has_conflicts": False,
                "risk_level": "LOW",
                "evidence_summary": f"Search error: {str(e)}",
                "method": "brave_zaubacorp_enhanced_error",
                "web_search_performed": False,
                "error": str(e)
            }


class RealIndianWebSearcher:
    """Streamlined OpenAI search + Brave API integration"""
    
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        openai.api_key = self.openai_api_key
        
        # Check OpenAI version
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.openai_api_key)
            self.use_new_client = True
        except ImportError:
            self.client = None
            self.use_new_client = False

    # @capture_span()
    def comprehensive_openai_search(self, company_name: str) -> Dict[str, any]:
        """Comprehensive OpenAI knowledge search"""
        
        try:
            prompt = f"""
You are an expert on Indian company registrations and the MCA (Ministry of Corporate Affairs) database.

Your task: Analyze if "{company_name}" is a registered Indian company based on your knowledge.

Consider these data sources in your analysis:
- ZaubaCorp.com (Indian corporate database)
- IndiaFilings.com (MCA registration platform)
- Tofler.in (Corporate information)
- Official MCA records
- Known major Indian companies

IMPORTANT GUIDELINES:
1. Only confirm if you have specific knowledge of this company being registered in India
2. Consider exact matches and very close phonetic matches
3. Look for variations like "Private Limited", "Pvt Ltd" suffixes
4. Consider major pharmaceutical, tech, automotive, and other industry companies
5. Be conservative - only flag if you're confident

Respond in this exact format:
COMPANY_FOUND: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
DATA_SOURCE: [Where you know this from - zaubacorp/known companies/etc]
COMPANY_DETAILS: [Any specific details you know - CIN, status, industry, etc]
REASONING: [Brief explanation of why you flagged it or didn't]

Examples:
- "Reliance Industries" → YES, HIGH confidence (major known company)
- "Infosis" → YES, MEDIUM confidence (phonetic match to Infosys)
- "Random Tech Solutions" → NO, LOW confidence (generic name, no knowledge)

Analyze: "{company_name}"
"""
            
            if self.use_new_client and self.client:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert on Indian corporate registrations. Be accurate and conservative in your assessments."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                content = response.choices[0].message.content
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert on Indian corporate registrations. Be accurate and conservative in your assessments."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                content = response.choices[0].message.content
            
            # Parse the structured response
            content_lower = content.lower()
            
            company_found = "company_found: yes" in content_lower
            high_confidence = "confidence: high" in content_lower
            medium_confidence = "confidence: medium" in content_lower
            
            return {
                "found_by_openai": company_found,
                "confidence_level": "HIGH" if high_confidence else "MEDIUM" if medium_confidence else "LOW",
                "full_response": content,
                "reasoning": content
            }
            
        except Exception as e:
            return {
                "found_by_openai": False,
                "confidence_level": "LOW",
                "full_response": f"Error: {str(e)}",
                "reasoning": "Search failed due to error"
            }

    # @capture_span()
    def comprehensive_web_check(self, company_name: str) -> Dict[str, any]:
        """Multi-tier search with Brave API integration"""
        
        # Tier 1: OpenAI knowledge search
        openai_result = self.comprehensive_openai_search(company_name)
        
        if openai_result.get("found_by_openai") and openai_result.get("confidence_level") in ["HIGH", "MEDIUM"]:
            return {
                "has_conflicts": True,
                "risk_level": "HIGH" if openai_result.get("confidence_level") == "HIGH" else "MEDIUM",
                "evidence_summary": f"OpenAI knowledge search found company with {openai_result.get('confidence_level')} confidence",
                "method": "openai_comprehensive_single_query",
                "web_search_performed": True,
                "search_method": "openai_knowledge_comprehensive",
                "tier_used": "tier_1_openai_only",
                "openai_analysis": openai_result
            }
        
        # Tier 2: Brave API ZaubaCorp search
        brave_api_key = os.environ.get("BRAVE_API_KEY")
        
        if not brave_api_key:
            return {
                "has_conflicts": False,
                "risk_level": "LOW",
                "evidence_summary": "No evidence found via OpenAI knowledge search. Brave API not configured.",
                "method": "openai_only_no_brave",
                "web_search_performed": True,
                "search_method": "openai_knowledge_only",
                "tier_used": "tier_1_only_no_brave_key",
                "openai_analysis": openai_result
            }
        
        try:
            brave_searcher = BraveZaubaCorpSearcher()
            brave_result = brave_searcher.comprehensive_zaubacorp_check(company_name)
            
            if brave_result.get("has_conflicts"):
                return {
                    "has_conflicts": True,
                    "risk_level": brave_result.get("risk_level", "HIGH"),
                    "evidence_summary": f"Found on ZaubaCorp via Brave API: {brave_result.get('evidence_summary', '')}",
                    "method": "brave_zaubacorp_after_openai_negative",
                    "web_search_performed": True,
                    "search_method": "openai_plus_brave_zaubacorp",
                    "tier_used": "tier_2_brave_zaubacorp",
                    "openai_analysis": openai_result,
                    "brave_analysis": brave_result.get("zaubacorp_analysis")
                }
            
        except Exception as e:
            pass
        
        # No evidence found
        return {
            "has_conflicts": False,
            "risk_level": "LOW",
            "evidence_summary": "No registration evidence found via OpenAI knowledge or ZaubaCorp search",
            "method": "comprehensive_multi_tier_negative",
            "web_search_performed": True,
            "search_method": "openai_plus_brave_comprehensive",
            "tier_used": "both_tiers_no_evidence",
            "openai_analysis": openai_result
        }


# ===============================
# UPDATED TRADEMARK VALIDATOR
# ===============================

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
        
        # FIXED: Separate location words from government words
        self.location_words = countries | states | {
            'asia', 'europe', 'africa', 'australia', 'antarctica', 'north america', 
            'south america'
        }
        
        # FIXED: Remove location words from government words
        self.government_words = {
            'board', 'federal', 'municipal', 'commission', 'republic', 'authority', 
            'president', 'statute', 'statutory', 'rashtrapati', 'prime minister', 
            'court', 'judiciary', 'chief minister', 'governor', 'union',
            'minister', 'central', 'nation'
            # REMOVED: 'national', 'bharat', 'indian' - these should only be location checks
        }
        
        self.suffixes = {
            'private limited', 'limited', 'llp', 'llc', 'pvt ltd', 'pvt. ltd.',
            'p. ltd', 'pte ltd', 'ltd'
        }

    # @capture_span()
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
        
    # @capture_span()
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

    # @capture_span()
    def check_location_name(self, wordmark: str) -> Dict[str, bool]:
        """FIXED: Only flag if the name is ONLY location words, not when combined with other words"""
        if not wordmark:
            return {'is_location': False, 'matched_locations': []}
            
        wordmark_lower = wordmark.lower().strip()
        matched_locations = []
        
        # Remove suffix if present
        base_name = wordmark_lower
        for suffix in self.suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[:-(len(suffix))].strip()
                break
        
        words = base_name.split()
        
        # SPECIAL CASE: Allow "India", "Indian", "Bharat" for Indian companies when combined with other words
        allowed_for_indian_companies = {'india', 'indian', 'bharat'}
        
        # Check each word
        location_word_count = 0
        non_location_word_count = 0
        
        for word in words:
            if word in self.location_words:
                location_word_count += 1
                matched_locations.append(word)
            elif word not in {'of', 'and', 'the', '&'}:  # Don't count joining words
                non_location_word_count += 1
        
        # NEW LOGIC: Only flag if ALL meaningful words are location words AND no other business words
        
        # Case 1: Single word that is a non-India location
        if len(words) == 1 and words[0] in self.location_words and words[0] not in allowed_for_indian_companies:
            return {'is_location': True, 'matched_locations': [words[0]]}
        
        # Case 2: Only location words (no business words) and contains non-India locations
        if non_location_word_count == 0 and location_word_count > 0:
            # Check if any non-India locations
            non_india_locations = [loc for loc in matched_locations if loc not in allowed_for_indian_companies]
            if non_india_locations:
                return {'is_location': True, 'matched_locations': non_india_locations}
        
        # Case 3: Mixed with business words - ALLOW (this is the key fix)
        if non_location_word_count > 0:
            return {'is_location': False, 'matched_locations': []}
        
        # Case 4: Only India/Indian/Bharat - ALLOW for Indian companies
        if location_word_count > 0 and all(loc in allowed_for_indian_companies for loc in matched_locations):
            return {'is_location': False, 'matched_locations': []}
        
        return {'is_location': False, 'matched_locations': []}

    # @capture_span()
    def check_government_patronage(self, wordmark: str) -> Dict[str, bool]:
        """FIXED: Check for government patronage implications - exclude location words"""
        words = wordmark.lower().split()
        restricted_matches = []
        
        for word in words:
            # FIXED: Only check actual government words, not location words
            if word in self.government_words:
                restricted_matches.append(word)
        
        return {
            'implies_patronage': len(restricted_matches) > 0,
            'restricted_words_found': restricted_matches
        }
        
    # @capture_span()
    def check_articles(self, wordmark: str) -> Dict[str, bool]:
        """Check if wordmark only contains articles/pronouns"""
        if not wordmark:
            return {'has_articles': False, 'restricted_words_found': []}
            
        words = wordmark.lower().split()
        
        if len(words) > 0:
            restricted_matches = []
            for word in words:
                if word in self.articles['articles']:
                    restricted_matches.append(word)
            
            # Only flag if ALL words are articles/pronouns
            all_words_are_articles = len(restricted_matches) == len(words)
            
            return {
                'has_articles': all_words_are_articles,
                'restricted_words_found': restricted_matches if all_words_are_articles else []
            }
        
        return {
            'has_articles': wordmark.lower() in self.articles['articles'],
            'restricted_words_found': [wordmark.lower()] if wordmark.lower() in self.articles['articles'] else []
        }
        
    # @capture_span()
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
        
    # @capture_span()
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
        
    # @capture_span()
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
        
    # @capture_span()
    def _generate_validation_messages(self, results: Dict) -> List[str]:
        """Generate human-readable validation messages"""
        messages = []
        
        if results['translation_check'].get('has_meaning'):
            messages.append(f"Warning: Wordmark has meaning in other languages")
            
        if results['location_check']['is_location']:
            messages.append(f"Invalid: Contains only location name(s): {', '.join(results['location_check']['matched_locations'])}")
            
        if results['government_check']['implies_patronage']:
            messages.append(f"Invalid: Uses restricted government terms: {', '.join(results['government_check']['restricted_words_found'])}")
            
        if results['company_similarity']['has_similarities']:
            messages.append("Invalid: Similar to existing company names")
            
        if results['embassy_check']['has_embassy_connection']:
            messages.append(f"Invalid: Suggests embassy/consulate connection: {', '.join(results['embassy_check']['matched_terms'])}")

        if results['articles_check']['has_articles']:
            messages.append(f"Invalid: Contains only articles/pronouns")
            
        return messages


# ===============================
# HELPER FUNCTIONS
# ===============================

def get_cache_key(text: str) -> str:
    """Generate a cache key for fallback requests"""
    return hashlib.md5(text.lower().encode()).hexdigest()

def is_cache_valid(timestamp: datetime) -> bool:
    """Check if cache entry is still valid"""
    return datetime.now() - timestamp < CACHE_DURATION

def check_known_exact_matches(company_name: str) -> List[str]:
    """Check against database of known companies for exact matches"""
    company_lower = company_name.lower().strip()
    
    # Remove common suffixes for matching
    for suffix in ['private limited', 'pvt ltd', 'ltd', 'llp', 'india', 'technologies', 'corporation', 'corp']:
        if company_lower.endswith(suffix):
            company_lower = company_lower.replace(suffix, '').strip()
    
    # Database of known companies (exact matches only)
    known_companies = {
        'reliance': 'Reliance Industries Limited',
        'infosys': 'Infosys Limited', 
        'tcs': 'Tata Consultancy Services',
        'wipro': 'Wipro Limited',
        'hcl': 'HCL Technologies',
        'zomato': 'Zomato Limited',
        'swiggy': 'Bundl Technologies Private Limited (Swiggy)',
        'flipkart': 'Flipkart Private Limited',
        'paytm': 'One97 Communications Limited',
        'ola': 'ANI Technologies Private Limited (Ola)',
        'phonepe': 'PhonePe Private Limited',
        'relians': 'Reliance Industries Limited',
        'infosis': 'Infosys Limited',
        'flipcart': 'Flipkart Private Limited',
    }
    
    exact_matches = []
    if company_lower in known_companies:
        exact_matches.append(known_companies[company_lower])
    
    return exact_matches

# @capture_span()
def perform_actual_web_search(company_name: str) -> Dict[str, any]:
    """Comprehensive web search using existing components"""
    try:
        # Check cache first
        cache_key = get_cache_key(f"web_search_{company_name}")
        if cache_key in fallback_cache:
            cached_entry = fallback_cache[cache_key]
            if is_cache_valid(cached_entry["timestamp"]):
                return cached_entry["data"]
        
        # Step 1: Check known exact matches (fast)
        exact_matches = check_known_exact_matches(company_name)
        if exact_matches:
            result = {
                "has_conflicts": True,
                "risk_level": "HIGH",
                "exact_matches": exact_matches,
                "similar_matches": [],
                "method": "known_exact_match"
            }
        else:
            # Step 2: Use RealIndianWebSearcher
            try:
                searcher = RealIndianWebSearcher()
                web_result = searcher.comprehensive_web_check(company_name)
                
                result = {
                    "has_conflicts": web_result.get("has_conflicts", False),
                    "risk_level": web_result.get("risk_level", "LOW"),
                    "exact_matches": [],
                    "similar_matches": [],
                    "method": web_result.get("method", "web_search"),
                    "evidence_summary": web_result.get("evidence_summary", "")
                }
                
                # Add matches if found
                if web_result.get("has_conflicts"):
                    result["exact_matches"].append(f"{company_name} (Found online)")
                    
            except Exception as e:
                result = {
                    "has_conflicts": False,
                    "risk_level": "LOW",
                    "exact_matches": [],
                    "similar_matches": [],
                    "method": "web_search_error"
                }
        
        # Cache result
        fallback_cache[cache_key] = {
            "data": result,
            "timestamp": datetime.now()
        }
        
        return result
        
    except Exception as e:
        return {
            "has_conflicts": False,
            "risk_level": "LOW",
            "exact_matches": [],
            "similar_matches": [],
            "method": "search_error"
        }

        
# @capture_span()
def get_phonetic_representation(word):
    word_lowercase = word.lower()
    primary, secondary = doublemetaphone.doublemetaphone(word_lowercase)
    return primary or secondary or word_lowercase
    
# @capture_span()
def get_embedding(text, model="text-embedding-3-small"):
    try:
        if not text or text.isspace():
            st.error("Please enter a valid company name that is not just a suffix (like 'private limited', 'ltd', etc.)")
            return None
        
        normalized_text = text.lower()
        
        try:
            # Try new OpenAI client format first
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.embeddings.create(
                model=model,
                input=[normalized_text]
            )
            return response.data[0].embedding
        except:
            # Fallback to legacy format
            response = openai.Embedding.create(
                model=model,
                input=[normalized_text]
            )
            return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        # Try fallback to older model if the new one fails
        try:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=[normalized_text]
                )
                return response.data[0].embedding
            except:
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=[normalized_text]
                )
                return response["data"][0]["embedding"]
        except Exception as e2:
            st.error(f"Fallback embedding also failed: {str(e2)}")
            return None

# @capture_span()
def calculate_phonetic_similarity(word1, word2):
    phonetic1 = get_phonetic_representation(word1)
    phonetic2 = get_phonetic_representation(word2)
    return SequenceMatcher(None, phonetic1, phonetic2).ratio()
    
# @capture_span()
def calculate_hybrid_score(phonetic_score, semantic_score, phonetic_weight=0.6, semantic_weight=0.4):
    return (phonetic_weight * phonetic_score) + (semantic_weight * semantic_score)

# @capture_span()
def check_multiple_phonetic_matches(wordmark, index, model="text-embedding-3-small", namespace=""):
    """SMART LOGIC: Trigger web search unless high risk database matches found"""
    try:
        # Initialize validator
        validator = TrademarkValidator()
        
        # Check if input is just a suffix
        cleaned_wordmark = validator.remove_suffix(wordmark)
        if not cleaned_wordmark:
            st.error("Please enter a valid company name that is not just a suffix (like 'private limited', 'ltd', etc.)")
            return None
              
        cleaned_wordmark = cleaned_wordmark.lower()
        input_embedding = get_embedding(cleaned_wordmark, model=model)

        if input_embedding is None:
            return None

        query_result = index.query(
            vector=input_embedding,
            top_k=10,
            include_metadata=True,
            namespace=namespace
        )

        matches = []
        for match in query_result["matches"]:
            # Handle different metadata formats between MCA and Trademark indexes
            if 'original_data' in match.get('metadata', {}):
                # MCA index format
                stored_wordmark = match["metadata"].get("original_data", "")
                stored_classes = None  # MCA doesn't have class info
            else:
                # Trademark index format
                stored_wordmark = match.get('metadata', {}).get('wordMark', '')
                stored_classes = match.get('metadata', {}).get('wclass', [])

            # Calculate similarity using cleaned names
            phonetic_score = calculate_phonetic_similarity(cleaned_wordmark, stored_wordmark.lower())
            semantic_score = match["score"]
            hybrid_score = calculate_hybrid_score(phonetic_score, semantic_score)

            matches.append({
                "Matching Wordmark": stored_wordmark,
                "Cleaned Wordmark": validator.remove_suffix(stored_wordmark),
                "Class": stored_classes,
                "Phonetic Score": phonetic_score,
                "Semantic Score": semantic_score,
                "Hybrid Score": hybrid_score,
                "Source": "Database"
            })

        # SMART THRESHOLD LOGIC: Categorize matches by risk level
        high_risk_matches = [
            match for match in matches 
            if match["Hybrid Score"] > 0.8 or match["Phonetic Score"] > 0.85
        ]
        
        medium_risk_matches = [
            match for match in matches 
            if (match["Hybrid Score"] > 0.65 or match["Phonetic Score"] > 0.75) 
            and match not in high_risk_matches
        ]
        
        low_risk_matches = [
            match for match in matches 
            if (match["Hybrid Score"] > 0.5 or match["Phonetic Score"] > 0.6) 
            and match not in high_risk_matches 
            and match not in medium_risk_matches
        ]
        
        # DECISION LOGIC:
        # - If HIGH RISK matches found → Skip web search (we already have strong evidence)
        # - If only MEDIUM/LOW risk matches → Trigger web search (need more validation)
        # - If no significant matches → Trigger web search
        
        should_trigger_web_search = len(high_risk_matches) == 0
        
        if should_trigger_web_search:
            try:
                with st.spinner("🌐 Performing comprehensive web search..."):
                    web_search_result = perform_actual_web_search(cleaned_wordmark)
                
                if web_search_result.get("has_conflicts"):
                    # Create synthetic matches from web search
                    for exact_match in web_search_result.get("exact_matches", []):
                        matches.append({
                            "Matching Wordmark": exact_match,
                            "Cleaned Wordmark": exact_match,
                            "Class": None,
                            "Phonetic Score": 1.0,
                            "Semantic Score": 0.95,
                            "Hybrid Score": 0.98,
                            "Source": "Web_Search_Exact"
                        })
                    
                    for similar_match in web_search_result.get("similar_matches", []):
                        risk_score = 0.85 if similar_match.get("risk_level") == "HIGH" else 0.75
                        matches.append({
                            "Matching Wordmark": similar_match.get("company_name", ""),
                            "Cleaned Wordmark": similar_match.get("company_name", ""),
                            "Class": None,
                            "Phonetic Score": risk_score,
                            "Semantic Score": risk_score,
                            "Hybrid Score": risk_score,
                            "Source": "Web_Search_Similar"
                        })
                    
                    st.info(f"🌐 Web search added {len(web_search_result.get('exact_matches', [])) + len(web_search_result.get('similar_matches', []))} additional matches")
                    
            except Exception as e:
                st.warning(f"Web search failed: {str(e)}")

        matches = sorted(matches, key=lambda x: x["Hybrid Score"], reverse=True)
        return matches
        
    except Exception as e:
        st.error(f"Error checking phonetic matches: {str(e)}")
        return None

# @capture_span()
def suggest_similar_names(wordmark):
    try:
        try:
            # Try new OpenAI client format first
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a creative naming assistant who generates unique and meaningful alternative names for businesses. Your goal is to provide exactly 5 unique name suggestions by modifying the input name using a prefix, suffix, or an additional word while ensuring that the essence of the original name remains intact.
                                    Understand the context of the input name before suggesting alternatives.
                                    IMPORTANT: Provide ONLY the name suggestions without any explanations or formatting, one per line."""
                    },
                    {
                        "role": "user",
                        "content": f"Suggest five creative and unique alternative names for the word '{wordmark}'. Only provide the names, no descriptions or explanations."
                    }
                ],
                max_tokens=150,
                temperature=0.9
            )
            response_text = response.choices[0].message.content.strip()
        except:
            # Fallback to legacy format
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a creative naming assistant who generates unique and meaningful alternative names for businesses. Your goal is to provide exactly 5 unique name suggestions by modifying the input name using a prefix, suffix, or an additional word while ensuring that the essence of the original name remains intact.
                                    Understand the context of the input name before suggesting alternatives.
                                    IMPORTANT: Provide ONLY the name suggestions without any explanations or formatting, one per line."""
                    },
                    {
                        "role": "user",
                        "content": f"Suggest five creative and unique alternative names for the word '{wordmark}'. Only provide the names, no descriptions or explanations."
                    }
                ],
                max_tokens=150,
                temperature=0.9
            )
            response_text = response.choices[0].message.content.strip()
        
        raw_suggestions = response_text.split("\n")
        cleaned_suggestions = []
        
        for suggestion in raw_suggestions:
            # Clean up suggestions
            cleaned = suggestion.strip()
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)  # Remove numbering
            cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Remove markdown
            if ' - ' in cleaned:
                cleaned = cleaned.split(' - ')[0].strip()
            if cleaned.startswith("Here are") or "suggestions" in cleaned.lower():
                continue
                
            if cleaned and len(cleaned) > 0:
                cleaned_suggestions.append(cleaned)
        
        return cleaned_suggestions[:5]
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")
        return []

# @capture_span()
def generate_validation_safe_suggestions(original_name: str, conflict_info: Dict) -> List[str]:
    """Generate name suggestions that avoid both MCA and trademark conflicts"""
    
    try:
        mca_conflicts = conflict_info.get("mca_conflicts", [])
        trademark_conflicts = conflict_info.get("trademark_conflicts", [])
        trademark_class = conflict_info.get("trademark_class")
        
        # Analyze conflict patterns
        conflicting_words = set()
        conflicting_patterns = []
        
        # Extract problematic words from MCA conflicts
        for conflict in mca_conflicts:
            company_name = conflict.get("Matching Wordmark", "")
            if company_name:
                words = company_name.lower().split()
                conflicting_words.update(words)
                conflicting_patterns.append(company_name.lower())
        
        # Extract problematic words from trademark conflicts  
        for conflict in trademark_conflicts:
            trademark_name = conflict.get("Matching Wordmark", "")
            if trademark_name:
                words = trademark_name.lower().split()
                conflicting_words.update(words)
                conflicting_patterns.append(trademark_name.lower())
        
        # Remove common/generic words that aren't really conflicts
        generic_words = {'private', 'limited', 'ltd', 'pvt', 'company', 'corp', 'inc', 'llp', 'the', 'and', 'of', 'for', 'with'}
        conflicting_words = conflicting_words - generic_words
        
        # Create enhanced prompt for smart suggestions
        prompt = f"""
Generate 5 unique, creative business name alternatives for "{original_name}" that:

1. AVOID these conflicting words/patterns: {list(conflicting_words)[:10]}
2. AVOID phonetic similarity to: {conflicting_patterns[:3]}
3. Maintain the business essence/industry context
4. Use different root words or creative combinations
5. Are professional and brandable

CONFLICT AVOIDANCE RULES:
- Don't use any words from the conflict list
- Change the core words, not just suffixes
- Use synonyms, creative combinations, or completely different approaches
- Make them sound distinct when spoken (different phonetics)

Original name: "{original_name}"
Industry context: {f"Trademark class {trademark_class}" if trademark_class else "General business"}

Provide ONLY the 5 names, one per line, no explanations:
"""

        # Generate suggestions using OpenAI
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a creative business naming expert. Generate unique names that avoid conflicts and sound professionally distinct."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.8  # Higher creativity
            )
            response_text = response.choices[0].message.content.strip()
        except:
            # Fallback to legacy format
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a creative business naming expert. Generate unique names that avoid conflicts and sound professionally distinct."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.8
            )
            response_text = response.choices[0].message.content.strip()
        
        # Parse and clean suggestions
        raw_suggestions = response_text.split("\n")
        cleaned_suggestions = []
        
        for suggestion in raw_suggestions:
            cleaned = suggestion.strip()
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)  # Remove numbering
            cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Remove markdown
            
            if cleaned and len(cleaned) > 2 and not any(word in cleaned.lower() for word in conflicting_words):
                cleaned_suggestions.append(cleaned)
        
        # Validate suggestions don't conflict (basic check)
        validated_suggestions = []
        for suggestion in cleaned_suggestions[:8]:  # Check more than needed
            suggestion_words = set(suggestion.lower().split())
            if not (suggestion_words & conflicting_words):  # No overlapping words
                validated_suggestions.append(suggestion)
                if len(validated_suggestions) >= 5:
                    break
        
        return validated_suggestions[:5]
        
    except Exception as e:
        st.error(f"Error generating smart suggestions: {str(e)}")
        # Fallback to simple suggestions
        return [
            f"New{original_name.split()[0]}Corp",
            f"Prime{original_name.split()[-1]}Solutions", 
            f"Alpha{original_name.replace(' ', '')}",
            f"NextGen{original_name.split()[0]}",
            f"Innovative{original_name.split()[-1]}"
        ]

# @capture_span()
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

                if hasattr(query_result, 'matches') and query_result.matches and query_result.matches[0].score >= similarity_threshold:
                    is_unique = False
                elif 'matches' in query_result and query_result['matches'] and query_result['matches'][0]['score'] >= similarity_threshold:
                    is_unique = False

            if is_unique:
                unique_suggestions.append(suggestion)
        return unique_suggestions
    except Exception as e:
        st.error(f"Error validating suggestions: {str(e)}")
        return []

# @capture_span()
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

# @capture_span()
def extract_class_number(trademark_class_text: str) -> str:
    """Extract the class number from GPT-4's classification response"""
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

# @capture_span()
def classify_objective_gpt4(objective: str):
    """Use GPT-4 to classify the business objective"""
    if not objective or len(objective) < 5:
        return "Invalid input. Please provide a meaningful objective."

    try:
        # Get embedding with proper error handling
        try:
            # Try new OpenAI client format first
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=objective.strip().lower()
            )
            query_embedding = response.data[0].embedding
        except:
            # Fallback to legacy format
            response = openai.Embedding.create(
                input=objective.strip().lower(),
                model="text-embedding-ada-002"
            )
            query_embedding = response["data"][0]["embedding"]

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
        
        Provide a detailed classification with the most appropriate class number.
        """

        try:
            # Try new OpenAI client format first
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            gpt_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a trademark classification assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return gpt_response.choices[0].message.content
        except:
            # Fallback to legacy format
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a trademark classification assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return gpt_response["choices"][0]["message"]["content"]
            
    except Exception as e:
        st.error(f"Error in trademark classification: {str(e)}")
        return "Classification failed due to error"

# @capture_span()
def check_class_specific_matches(wordmark: str, trademark_class: str):
    """Check for similar marks specifically within the identified trademark class"""
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

# @capture_span()
def check_multiple_phonetic_matches_db_only(wordmark, index, model="text-embedding-3-small", namespace=""):
    """Database-only search (NO web search) for trademark validation"""
    try:
        validator = TrademarkValidator()
        
        cleaned_wordmark = validator.remove_suffix(wordmark)
        if not cleaned_wordmark:
            return None
              
        cleaned_wordmark = cleaned_wordmark.lower()
        input_embedding = get_embedding(cleaned_wordmark, model=model)

        if input_embedding is None:
            return None

        query_result = index.query(
            vector=input_embedding,
            top_k=20,  # Fewer results for cross-class trademark search
            include_metadata=True,
            namespace=namespace
        )

        matches = []
        for match in query_result["matches"]:
            # Trademark index format only
            stored_wordmark = match.get('metadata', {}).get('wordMark', '')
            stored_classes = match.get('metadata', {}).get('wclass', [])

            # Calculate similarity using cleaned names
            phonetic_score = calculate_phonetic_similarity(cleaned_wordmark, stored_wordmark.lower())
            semantic_score = match["score"]
            hybrid_score = calculate_hybrid_score(phonetic_score, semantic_score)

            matches.append({
                "Matching Wordmark": stored_wordmark,
                "Cleaned Wordmark": validator.remove_suffix(stored_wordmark),
                "Class": stored_classes,
                "Phonetic Score": phonetic_score,
                "Semantic Score": semantic_score,
                "Hybrid Score": hybrid_score,
                "Source": "Trademark_Database"
            })

        # NO WEB SEARCH - Database only for trademark validation
        matches = sorted(matches, key=lambda x: x["Hybrid Score"], reverse=True)
        return matches
        
    except Exception as e:
        st.error(f"Error checking trademark database: {str(e)}")
        return None

# @capture_span()
def classify_and_check_trademark_db_only(objective: str, wordmark: str, trademark_class: str):
    """Trademark validation using database ONLY (no web search)"""
    try:
        # Check within specific class (database only)
        class_specific_matches = check_class_specific_matches(wordmark, trademark_class)
        
        # Check across all classes (database only) - FIXED: Use DB-only function
        cross_class_matches = check_multiple_phonetic_matches_db_only(wordmark, trademark_index, model="text-embedding-3-small")
        
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
                if match["Hybrid Score"] > 0.7 or match["Phonetic Score"] > 0.8
            ]
            
        # Process cross-class matches (higher threshold since different class)
        if cross_class_matches:
            results['cross_class_matches'] = [
                match for match in cross_class_matches 
                if match["Hybrid Score"] > 0.75 or match["Phonetic Score"] > 0.85
            ]
            
        # Identify high-risk matches
        all_matches = results['class_matches'] + results['cross_class_matches']
        high_risk_matches = [
            match for match in all_matches
            if match["Hybrid Score"] > 0.8 or match["Phonetic Score"] > 0.85
        ]
        
        results['high_risk_matches'] = high_risk_matches
        results['suggestions_needed'] = bool(high_risk_matches)
        
        return results
        
    except Exception as e:
        st.error(f"Error in trademark database validation: {str(e)}")
        return None
    
# @capture_span()
def main():
    st.title("🏢 Enhanced MCA and Trademark Validation")
    st.write("Advanced validation tool with comprehensive web search integration for Indian companies.")

    # Show configuration status
    with st.expander("🔧 Configuration Status"):
        config_info = {}
        config_info["OpenAI"] = "✅ Configured" if os.environ.get("OPENAI_API_KEY") else "❌ Not configured"
        config_info["Pinecone"] = "✅ Configured" if os.environ.get("PINECONE_API_KEY") else "❌ Not configured"
        config_info["Brave API"] = "✅ Configured" if os.environ.get("BRAVE_API_KEY") else "⚠️ Not configured (web search limited)"
        
        for service, status in config_info.items():
            st.write(f"**{service}:** {status}")

    validator = TrademarkValidator()
    
    col1, col2 = st.columns(2)
    with col1:
        wordmark = st.text_input("🏷️ Enter the Name/Wordmark:", "")
    with col2:
        objective = st.text_input("🎯 Enter the Business Objective:", "")

    if st.button("🔍 Validate Name", type="primary"):
        if not wordmark:
            st.warning("Please enter the Name/Wordmark.")
            return
            
        # First check if input is just a suffix
        cleaned_name = validator.remove_suffix(wordmark)
        if not cleaned_name:
            st.error("Please enter a valid name. The input cannot be just a suffix (like 'private limited', 'ltd', etc.)")
            return

        # Create tabs for MCA and Trademark validation results
        mca_tab, trademark_tab, suggestions_tab = st.tabs(["🏢 MCA Validation", "™️ Trademark Validation", "💡 Suggestions"])
        
        with mca_tab:
            st.header("🏢 MCA Name Validation")
            st.info("Checking against MCA database and performing comprehensive web search for Indian companies")
            
            # Perform MCA name validation
            with st.spinner("Performing enhanced MCA name validation..."):
                validation_results = validator.validate_trademark(wordmark)
                
                # Get the correct namespace for MCA
                try:
                    index_stats = mca_index.describe_index_stats()
                    namespaces = list(index_stats.get('namespaces', {}).keys())
                    namespace_to_use = namespaces[0] if namespaces else ""
                except Exception as e:
                    st.error(f"Failed to connect to Pinecone MCA index: {str(e)}")
                    namespace_to_use = ""
                    
                # Check for similar names in MCA database + web search
                mca_matches = check_multiple_phonetic_matches(wordmark, mca_index, namespace=namespace_to_use)
            
            # Analyze matches by source and risk level
            if mca_matches:
                db_matches = [m for m in mca_matches if m.get("Source") == "Database"]
                web_matches = [m for m in mca_matches if m.get("Source") in ["Web_Search_Exact", "Web_Search_Similar"]]
                
                # Categorize by risk level
                high_risk_matches = [m for m in mca_matches if m["Hybrid Score"] > 0.8 or m["Phonetic Score"] > 0.85]
                medium_risk_matches = [m for m in mca_matches if (m["Hybrid Score"] > 0.65 or m["Phonetic Score"] > 0.75) and m not in high_risk_matches]
                
                st.info(f"📊 **Search Summary:** Found {len(db_matches)} database matches and {len(web_matches)} web search matches")
                
                if web_matches:
                    st.success("🌐 **Web search was triggered** - comprehensive validation performed!")
                
            # Display MCA validation results
            if not validation_results['overall_validity']['is_valid']:
                st.error("### ⚠️ MCA Validation Issues Found!")
                for message in validation_results['overall_validity']['validation_messages']:
                    st.warning(f"• {message}")
                        
                if validation_results['translation_check'].get('has_meaning'):
                    with st.expander("🌐 Translation Details"):
                        if validation_results['translation_check'].get('english_meaning'):
                            st.write(f"**English meaning:** {validation_results['translation_check']['english_meaning']}")
                        if validation_results['translation_check'].get('hindi_meaning'):
                            st.write(f"**Hindi meaning:** {validation_results['translation_check']['hindi_meaning']}")
            
            # Display MCA database + web search match results
            if mca_matches:
                # Show high-risk matches first
                if high_risk_matches:
                    st.error("### 🚨 High Risk MCA Name Matches Found!")
                    for match in high_risk_matches:
                        source_icon = "🌐" if match.get("Source") in ["Web_Search_Exact", "Web_Search_Similar"] else "💾"
                        with st.expander(f"{source_icon} **HIGH RISK:** {match['Matching Wordmark']}"):
                            st.write(f"- **Source:** {match.get('Source', 'Database')}")
                            st.write(f"- **Phonetic Score:** {match['Phonetic Score']:.3f}")
                            st.write(f"- **Semantic Score:** {match['Semantic Score']:.3f}")
                            st.write(f"- **Hybrid Score:** {match['Hybrid Score']:.3f}")
                            if match.get("Source") in ["Web_Search_Exact", "Web_Search_Similar"]:
                                st.info("🌐 This match was found through comprehensive web search")

                # Show medium-risk matches
                elif medium_risk_matches:
                    st.warning("### ⚠️ Medium Risk MCA Names Found")
                    for match in medium_risk_matches:
                        source_icon = "🌐" if match.get("Source") in ["Web_Search_Exact", "Web_Search_Similar"] else "💾"
                        with st.expander(f"{source_icon} **MEDIUM RISK:** {match['Matching Wordmark']}"):
                            st.write(f"- **Source:** {match.get('Source', 'Database')}")
                            st.write(f"- **Phonetic Score:** {match['Phonetic Score']:.3f}")
                            st.write(f"- **Semantic Score:** {match['Semantic Score']:.3f}")
                            st.write(f"- **Hybrid Score:** {match['Hybrid Score']:.3f}")
                            if match.get("Source") in ["Web_Search_Exact", "Web_Search_Similar"]:
                                st.info("🌐 This match was found through comprehensive web search")
                else:
                    st.success("✅ No high or medium risk MCA name matches found!")
                    
                    # Show low-risk matches in expandable section
                    low_risk_matches = [m for m in mca_matches if m not in high_risk_matches and m not in medium_risk_matches]
                    if low_risk_matches:
                        with st.expander(f"📋 View {len(low_risk_matches)} Low Risk Matches"):
                            for match in low_risk_matches[:5]:  # Show top 5 low risk
                                source_icon = "🌐" if match.get("Source") in ["Web_Search_Exact", "Web_Search_Similar"] else "💾"
                                st.write(f"{source_icon} {match['Matching Wordmark']} (Score: {match['Hybrid Score']:.3f})")
            else:
                if validation_results['overall_validity']['is_valid']:
                    st.success("✅ **MCA Validation Passed!** No issues found with the name.")
                else:
                    st.info("No similar MCA names found in database or web search, but other validation issues exist.")
                
            # Display detailed MCA validation report
            with st.expander("📋 View Detailed MCA Validation Report"):
                st.write("### Validation Checks Performed:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**1. Translation Check:**")
                    if validation_results['translation_check'].get('has_meaning'):
                        st.write("- ⚠️ Has meaning in other languages")
                        if validation_results['translation_check'].get('english_meaning'):
                            st.write(f"- English: {validation_results['translation_check']['english_meaning']}")
                        if validation_results['translation_check'].get('hindi_meaning'):
                            st.write(f"- Hindi: {validation_results['translation_check']['hindi_meaning']}")
                    else:
                        st.write("- ✅ No significant translations found")
                    
                    st.write("**2. Location Name Check:**")
                    if validation_results['location_check']['is_location']:
                        st.write(f"- ❌ Contains location names: {', '.join(validation_results['location_check']['matched_locations'])}")
                    else:
                        st.write("- ✅ No problematic location names found")
                    
                    st.write("**3. Government Terms Check:**")
                    if validation_results['government_check']['implies_patronage']:
                        st.write(f"- ❌ Contains restricted terms: {', '.join(validation_results['government_check']['restricted_words_found'])}")
                    else:
                        st.write("- ✅ No restricted government terms found")
                
                with col2:
                    st.write("**4. Embassy/Consulate Check:**")
                    if validation_results['embassy_check']['has_embassy_connection']:
                        st.write(f"- ❌ Contains embassy-related terms: {', '.join(validation_results['embassy_check']['matched_terms'])}")
                    else:
                        st.write("- ✅ No embassy/consulate connections found")

                    st.write("**5. Articles/Pronouns Check:**")
                    if validation_results['articles_check']['has_articles']:
                        st.write("- ❌ Contains only articles/pronouns")
                    else:
                        st.write("- ✅ No articles/pronouns issues found")
                        
                    st.write("**6. Database + Web Search:**")
                    if mca_matches:
                        db_count = len([m for m in mca_matches if m.get("Source") == "Database"])
                        web_count = len([m for m in mca_matches if m.get("Source") in ["Web_Search_Exact", "Web_Search_Similar"]])
                        st.write(f"- 💾 Database matches: {db_count}")
                        st.write(f"- 🌐 Web search matches: {web_count}")
                        if web_count > 0:
                            st.write("- ✅ Comprehensive web search performed")
                    else:
                        st.write("- ✅ No matches found in database or web")
                
        with trademark_tab:
            st.header("™️ Trademark Validation")
            st.info("Checking against trademark database (database search only)")
            
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
                        st.success("### 🎯 Classification Result")
                        with st.expander("View Full Classification Analysis"):
                            st.write(trademark_class_text)
                        st.info(f"**📋 Mapped Trademark Class: {class_number}**")

                        # Step 2: Check for similar trademarks (database only)
                        with st.spinner("Checking trademark database..."):
                            trademark_results = classify_and_check_trademark_db_only(objective, wordmark, class_number)

                        if trademark_results:
                            # Display results based on risk level
                            if trademark_results['high_risk_matches']:
                                st.error("### 🚨 High Risk Trademark Matches Found!")
                                for match in trademark_results['high_risk_matches']:
                                    with st.expander(f"**HIGH RISK:** {match['Matching Wordmark']}"):
                                        st.write(f"- **Class:** {match.get('Class', 'N/A')}")
                                        st.write(f"- **Phonetic Score:** {match['Phonetic Score']:.3f}")
                                        st.write(f"- **Semantic Score:** {match['Semantic Score']:.3f}")
                                        st.write(f"- **Hybrid Score:** {match['Hybrid Score']:.3f}")
                                        st.write(f"- **Same Class:** {'Yes' if match.get('Is Same Class') else 'No'}")
                            elif trademark_results['class_matches']:
                                st.warning("### ⚠️ Similar Trademarks Found in Same Class")
                                for match in trademark_results['class_matches']:
                                    with st.expander(f"**SAME CLASS:** {match['Matching Wordmark']}"):
                                        st.write(f"- **Class:** {match['Class']}")
                                        st.write(f"- **Phonetic Score:** {match['Phonetic Score']:.3f}")
                                        st.write(f"- **Semantic Score:** {match['Semantic Score']:.3f}")
                                        st.write(f"- **Hybrid Score:** {match['Hybrid Score']:.3f}")
                            elif trademark_results['cross_class_matches']:
                                st.info("### ℹ️ Similar Trademarks Found in Other Classes")
                                for match in trademark_results['cross_class_matches'][:3]:  # Show top 3
                                    with st.expander(f"**OTHER CLASS:** {match['Matching Wordmark']}"):
                                        st.write(f"- **Class:** {match['Class']}")
                                        st.write(f"- **Phonetic Score:** {match['Phonetic Score']:.3f}")
                                        st.write(f"- **Semantic Score:** {match['Semantic Score']:.3f}")
                                        st.write(f"- **Hybrid Score:** {match['Hybrid Score']:.3f}")
                            else:
                                st.success("✅ **No high-risk trademark matches found!**")
                        else:
                            st.info("No trademark validation results available.")

        with suggestions_tab:
            st.header("💡 Alternative Name Suggestions")
            
            # Determine if we need to generate suggestions
            need_suggestions = False
            suggestion_reasons = []
            
            # Check MCA validation results
            mca_has_conflicts = not validation_results['overall_validity']['is_valid']
            if mca_has_conflicts:
                need_suggestions = True
                suggestion_reasons.append("MCA validation issues found")
                
            # Check for high-risk MCA matches
            if mca_matches:
                high_risk_mca = [m for m in mca_matches if m["Hybrid Score"] > 0.8 or m["Phonetic Score"] > 0.85]
                if high_risk_mca:
                    need_suggestions = True
                    suggestion_reasons.append(f"Found {len(high_risk_mca)} high-risk MCA matches")
            
            # Check trademark validation results (if objective was provided)
            trademark_has_conflicts = False
            if objective and class_number and trademark_results:
                if trademark_results['suggestions_needed']:
                    need_suggestions = True
                    trademark_has_conflicts = True
                    suggestion_reasons.append("High-risk trademark matches found")
            
            if need_suggestions:
                st.warning(f"**Suggestions needed due to:** {', '.join(suggestion_reasons)}")
                
                with st.spinner("Generating conflict-aware alternative name suggestions..."):
                    # Gather conflict information for smart suggestions
                    all_mca_risk_matches = []
                    if mca_matches:
                        all_mca_risk_matches = [m for m in mca_matches if m["Hybrid Score"] > 0.6 or m["Phonetic Score"] > 0.7]
                    
                    conflict_info = {
                        "mca_conflicts": all_mca_risk_matches,
                        "trademark_conflicts": trademark_results.get("high_risk_matches", []) if trademark_results else [],
                        "original_name": cleaned_name,
                        "trademark_class": class_number if class_number else None
                    }
                    
                    # Generate smart suggestions that avoid conflicts
                    smart_suggestions = generate_validation_safe_suggestions(cleaned_name, conflict_info)

                    if smart_suggestions:
                        st.success("### ✅ **Validated Alternative Suggestions:**")
                        st.info("These suggestions are related to your input but designed to avoid conflicts")
                        
                        # Display suggestions with validation status
                        for i, suggestion in enumerate(smart_suggestions, 1):
                            with st.expander(f"**{i}.** {suggestion}", expanded=False):
                                # Show why this suggestion is different
                                original_words = set(cleaned_name.lower().split())
                                suggestion_words = set(suggestion.lower().split())
                                overlap = original_words & suggestion_words
                                different = suggestion_words - original_words
                                
                                if overlap:
                                    st.write(f"**Shared concept:** {', '.join(overlap)}")
                                if different:
                                    st.write(f"**New elements:** {', '.join(different)}")
                                
                                st.success("✅ **Pre-validated:** Passed MCA naming rules")
                                
                                # Quick validate button for each suggestion
                                if st.button(f"🔍 Quick Check", key=f"validate_{i}"):
                                    with st.spinner(f"Validating {suggestion}..."):
                                        # Quick validation check
                                        quick_validation = validator.validate_trademark(suggestion)
                                        if quick_validation['overall_validity']['is_valid']:
                                            st.success("✅ Suggestion passed quick validation!")
                                        else:
                                            st.warning("⚠️ Some validation issues found:")
                                            for msg in quick_validation['overall_validity']['validation_messages']:
                                                st.write(f"• {msg}")
                                
                        # Add note about further validation
                        st.info("""
                        💡 **Next Steps:** 
                        - These suggestions have been pre-validated for basic naming rules
                        - For final selection, run a full validation check on your chosen name
                        - Consider trademark registration search for your selected name
                        """)
                        
                    else:
                        st.warning("Could not generate suitable validated suggestions.")
                        
                        # Show manual suggestion guidelines
                        with st.expander("📋 Manual Suggestion Guidelines"):
                            st.write("**To create your own alternatives, try:**")
                            st.write("• Replace conflicting words with synonyms")
                            st.write("• Add professional prefixes: Pro, Smart, Digital, Modern")
                            st.write("• Use different suffixes: Hub, Works, Studio, Labs")
                            st.write("• Combine with industry terms relevant to your business")
                            
                            if conflict_info.get("mca_conflicts"):
                                conflicting_names = [m.get("Matching Wordmark", "") for m in conflict_info["mca_conflicts"][:3]]
                                st.write(f"**Avoid similarity to:** {', '.join(conflicting_names)}")
                            
            else:
                st.success("✅ **No suggestions needed - current name appears to be valid!**")
                st.balloons()
                
                # Show summary of why it's valid
                st.info("**Validation Summary:**")
                if validation_results['overall_validity']['is_valid']:
                    st.write("• ✅ Passed all MCA validation checks")
                if not mca_matches or not any(m["Hybrid Score"] > 0.8 for m in mca_matches):
                    st.write("• ✅ No high-risk MCA database or web matches")
                if not trademark_has_conflicts:
                    st.write("• ✅ No high-risk trademark conflicts found")

    # Add footer with information
    st.markdown("---")
    st.markdown("""
    ### 🔍 **About This Enhanced Validation Tool**
    
    **MCA Validation Features:**
    - ✅ Database search against MCA records
    - 🌐 Comprehensive web search integration (ZaubaCorp, known companies)
    - 🧠 Smart threshold logic - triggers web search for low/medium risk matches
    - 📊 Risk categorization (HIGH/MEDIUM/LOW)
    
    **Trademark Validation Features:**
    - 🎯 Automatic business objective classification
    - 📋 Class-specific trademark search
    - ⚖️ Cross-class similarity analysis
    
    **Smart Suggestions:**
    - 💡 Conflict-aware name generation
    - 🚫 Avoids known problematic words/patterns
    - 🎨 Maintains business context and essence
    """)

if __name__ == "__main__":
    main()
