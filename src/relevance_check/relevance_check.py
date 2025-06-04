from typing import List, Dict, Any, Tuple
from helpers.document_retriever import search_qa
from llm_factory.gemini import GoogleGen
from sentence_transformers import CrossEncoder, SentenceTransformer
from langchain_core.messages import HumanMessage
import numpy as np
from collections import defaultdict
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryStructure:
    def __init__(self, main_topic: str, explanation_level: str = "standard", 
                 target_audience: str = "general", filters: Dict[str, Any] = None):
        self.main_topic = main_topic
        self.explanation_level = explanation_level
        self.target_audience = target_audience
        self.filters = filters or {}

class CacheEntry:
    def __init__(self, judgment: Dict[str, Any], timestamp: float):
        self.judgment = judgment
        self.timestamp = timestamp

class HybridRelevanceChecker:
    def __init__(self):
        self.llm = GoogleGen()
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
        self._judgment_cache: Dict[str, CacheEntry] = {}
        self.cache_ttl = 3600
        self.judge_threshold = 0.7
        self.similarity_threshold = 0.75
        self.entailment_threshold = 0.8
        self.weights = {
            'llm_judge': 0.5,
            'similarity': 0.3,
            'entailment': 0.2
        }

    def _structure_query(self, query: str) -> QueryStructure:
        prompt = f"""Analyze this medical question and break it down into structured components. Return your analysis as a JSON object with the following fields:
- main_topic: The primary medical topic or condition being asked about
- explanation_level: One of ["basic", "standard", "detailed", "child-friendly"]
- target_audience: One of ["general", "patient", "medical_professional", "child"]
- filters: A dictionary of any specific requirements (e.g. {{"simplified_language": true}})

Question to analyze: {query}

Respond ONLY with the JSON object, no other text. Example format:
{{
    "main_topic": "cancer diagnosis",
    "explanation_level": "standard",
    "target_audience": "general",
    "filters": {{}}
}}"""
        
        try:
            response = self.llm([HumanMessage(content=prompt)])
            answer_text = response.content.strip()
            
            if '```json' in answer_text:
                answer_text = answer_text.split('```json')[1]
            if '```' in answer_text:
                answer_text = answer_text.split('```')[0]
            answer_text = answer_text.strip()
            
            # Add a check for empty answer_text or invalid JSON
            if not answer_text:
                logger.warning(f"LLM returned empty response for structuring query: \"{query}\"")
                return QueryStructure(query) # Return default structure
                
            try:
                structure = json.loads(answer_text)
                return QueryStructure(
                    main_topic=structure.get('main_topic', query),
                    explanation_level=structure.get('explanation_level', 'standard'),
                    target_audience=structure.get('target_audience', 'general'),
                    filters=structure.get('filters', {})
                )
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from LLM response for query \"{query}\": {e}. Raw response: \"{answer_text}\"")
                # Fallback to basic query structure if JSON parsing fails
                if "like I'm 5" in query.lower() or "explain to a child" in query.lower():
                    return QueryStructure(
                        main_topic=query.split("like I'm 5")[0].strip() if "like I'm 5" in query.lower()
                        else query.split("explain to a child")[0].strip(),
                        explanation_level="child-friendly",
                        target_audience="child",
                        filters={"simplified_language": True, "avoid_technical_terms": True}
                    )
                return QueryStructure(query)

        except Exception as e:
            logger.error(f"Error structuring query: {e}")
            if "like I'm 5" in query.lower() or "explain to a child" in query.lower():
                return QueryStructure(
                    main_topic=query.split("like I'm 5")[0].strip() if "like I'm 5" in query.lower()
                    else query.split("explain to a child")[0].strip(),
                    explanation_level="child-friendly",
                    target_audience="child",
                    filters={"simplified_language": True, "avoid_technical_terms": True}
                )
            return QueryStructure(query)

    def llm_judge(self, query: str, answer: str) -> Dict[str, Any]:
        cache_key = f"{query}|{answer}"
        current_time = time.time()
        
        # Clean expired cache
        expired_keys = [k for k, entry in self._judgment_cache.items() 
                      if current_time - entry.timestamp > self.cache_ttl]
        for key in expired_keys:
            del self._judgment_cache[key]
            
        if cache_key in self._judgment_cache:
            return self._judgment_cache[cache_key].judgment

        prompt = f"""You are a medical QA evaluation system..."""
        
        try:
            response = self.llm([HumanMessage(content=prompt)])
            answer_text = response.content.strip()
            judgment = json.loads(answer_text)
            
            # Validate and normalize
            judgment['confidence'] = float(judgment['confidence'])
            if not 0 <= judgment['confidence'] <= 1:
                judgment['confidence'] = max(0.0, min(1.0, judgment['confidence']))
            
            self._judgment_cache[cache_key] = CacheEntry(judgment, current_time)
            return judgment
        except Exception as e:
            logger.error(f"Error in LLM judgment: {e}")
            return {
                "judgment": "irrelevant",
                "confidence": 0.0,
                "reason": f"Error in judgment: {str(e)}"
            }

    def check_match(self, query: str) -> Dict[str, Any]:
        try:
            query_structure = self._structure_query(query)
            rag_results = search_qa(query=query_structure.main_topic, k=5)
            
            if not rag_results:
                return {'status': 'no_match', 'match_data': None}
            
            if query_structure.explanation_level == "child-friendly":
                rag_results = [
                    r for r in rag_results 
                    if any(term in r.get('answer', '').lower() 
                          for term in ['simple', 'easy to understand', 'like a', 'similar to'])
                ]
            
            best_eval = None
            best_score = -1
            
            for result in rag_results:
                try:
                    similarity = np.dot(
                        self.similarity_model.encode(query),
                        self.similarity_model.encode(result['question'])
                    ) / (
                        np.linalg.norm(self.similarity_model.encode(query)) *
                        np.linalg.norm(self.similarity_model.encode(result['question']))
                    )
                    
                    llm_judgment = self.llm_judge(query, result['answer'])
                    
                    score = (
                        llm_judgment['confidence'] * self.weights['llm_judge'] +
                        similarity * self.weights['similarity']
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_eval = {
                            'candidate': result,
                            'llm_judge': llm_judgment,
                            'similarity': similarity,
                            'combined_score': score
                        }
                except Exception as e:
                    logger.error(f"Error evaluating result: {e}")
                    continue
            
            if not best_eval:
                return {'status': 'no_match', 'match_data': None}
            
            if (best_eval['llm_judge']['judgment'] == 'relevant' and 
                best_eval['llm_judge']['confidence'] >= self.judge_threshold):
                status = 'relevant'
            elif best_eval['combined_score'] >= 0.6:
                status = 'partial'
            else:
                status = 'no_match'
            
            return {
                'status': status,
                'match_data': {
                    'answer': best_eval['candidate']['answer'],
                    'question': best_eval['candidate']['question'],
                    'confidence': best_eval['combined_score'],
                    'metrics': {
                        'llm_judge': best_eval['llm_judge'],
                        'similarity': best_eval['similarity']
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error in check_match: {e}")
            return {'status': 'no_match', 'match_data': None}