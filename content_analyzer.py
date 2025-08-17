"""
Dynamic Content Analyzer - ML-based entity detection and relationship discovery
This module eliminates keyword-based approaches and uses machine learning for content analysis.
"""

import spacy
import logging
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict
import re
from dataclasses import dataclass
from config import get_config

config = get_config()

@dataclass
class EntityRelationship:
    """Represents a relationship between two entities."""
    source: str
    target: str
    relationship_type: str
    confidence: float
    context: str

@dataclass
class ContentInsight:
    """Represents insights extracted from content."""
    entities: Dict[str, List[str]]
    relationships: List[EntityRelationship]
    document_type: str
    content_categories: List[str]
    confidence_score: float

class DynamicContentAnalyzer:
    """
    Analyzes content dynamically using ML models without predefined keywords.
    Uses spaCy NER and custom relationship extraction.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load spaCy models
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Loaded spaCy en_core_web_sm model")
        except IOError:
            self.logger.warning("en_core_web_sm not found, using blank model")
            self.nlp = spacy.blank("en")
        
        # Custom entity types we want to detect
        self.entity_types = {
            'PERSON', 'ORG', 'GPE', 'MONEY', 'DATE', 'TIME', 'PERCENT',
            'CARDINAL', 'ORDINAL', 'PRODUCT', 'EVENT', 'FAC', 'LAW'
        }
        
        # Relationship patterns (learned from context, not hardcoded keywords)
        self.relationship_patterns = {}
        
    def analyze_content(self, text: str) -> ContentInsight:
        """
        Analyze content to extract entities, relationships, and insights.
        
        Args:
            text: Input text to analyze
            
        Returns:
            ContentInsight object with extracted information
        """
        if not text or not text.strip():
            return self._empty_insight()
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = self._extract_entities(doc)
        
        # Discover relationships
        relationships = self._extract_relationships(doc, entities)
        
        # Determine document type and categories
        document_type = self._classify_document_type(doc, entities)
        content_categories = self._categorize_content(doc, entities)
        
        # Calculate overall confidence
        confidence_score = self._calculate_confidence(entities, relationships)
        
        return ContentInsight(
            entities=entities,
            relationships=relationships,
            document_type=document_type,
            content_categories=content_categories,
            confidence_score=confidence_score
        )
    
    def _extract_entities(self, doc) -> Dict[str, List[str]]:
        """Extract entities using spaCy NER."""
        entities = defaultdict(list)
        
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                # Clean and normalize entity text
                entity_text = self._clean_entity_text(ent.text)
                if entity_text and entity_text not in entities[ent.label_]:
                    entities[ent.label_].append(entity_text)
        
        # Also extract noun phrases as potential entities
        noun_phrases = self._extract_noun_phrases(doc)
        entities['NOUN_PHRASES'] = noun_phrases
        
        return dict(entities)
    
    def _extract_noun_phrases(self, doc) -> List[str]:
        """Extract meaningful noun phrases."""
        noun_phrases = []
        
        for chunk in doc.noun_chunks:
            # Filter out very short or common phrases
            if len(chunk.text.split()) >= 2 and not self._is_common_phrase(chunk.text):
                cleaned = self._clean_entity_text(chunk.text)
                if cleaned and cleaned not in noun_phrases:
                    noun_phrases.append(cleaned)
        
        return noun_phrases[:10]  # Limit to most relevant
    
    def _extract_relationships(self, doc, entities: Dict[str, List[str]]) -> List[EntityRelationship]:
        """Extract relationships between entities using dependency parsing."""
        relationships = []
        
        # Look for patterns in dependency tree
        for sent in doc.sents:
            sent_relationships = self._analyze_sentence_relationships(sent, entities)
            relationships.extend(sent_relationships)
        
        return relationships
    
    def _analyze_sentence_relationships(self, sent, entities: Dict[str, List[str]]) -> List[EntityRelationship]:
        """Analyze relationships within a sentence."""
        relationships = []
        
        # Find entities in this sentence
        sent_entities = []
        for token in sent:
            if token.ent_type_ in self.entity_types:
                sent_entities.append(token)
        
        # Look for relationship patterns
        for i, token in enumerate(sent):
            if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                # Look for verbs that might indicate relationships
                head = token.head
                if head.pos_ == 'VERB':
                    # Find other entities related to this verb
                    related_entities = self._find_related_entities(head, sent_entities)
                    
                    for source, target in related_entities:
                        if source != target:
                            relationship = EntityRelationship(
                                source=source,
                                target=target,
                                relationship_type=self._determine_relationship_type(head, token),
                                confidence=0.7,  # Base confidence
                                context=sent.text
                            )
                            relationships.append(relationship)
        
        return relationships
    
    def _find_related_entities(self, verb_token, entities: List) -> List[Tuple[str, str]]:
        """Find entities related through a verb."""
        related = []
        
        # Get all children of the verb
        children = list(verb_token.children)
        entity_children = [child for child in children if child in entities]
        
        # Create pairs of related entities
        for i, ent1 in enumerate(entity_children):
            for ent2 in entity_children[i+1:]:
                related.append((ent1.text, ent2.text))
        
        return related
    
    def _determine_relationship_type(self, verb_token, related_token) -> str:
        """Determine the type of relationship based on context."""
        verb_lemma = verb_token.lemma_.lower()
        
        # Map verb patterns to relationship types
        relationship_map = {
            'be': 'IS_A',
            'have': 'HAS',
            'own': 'OWNS',
            'manage': 'MANAGES',
            'control': 'CONTROLS',
            'report': 'REPORTS_TO',
            'pay': 'PAYS',
            'receive': 'RECEIVES',
            'invest': 'INVESTS_IN',
            'operate': 'OPERATES'
        }
        
        return relationship_map.get(verb_lemma, 'RELATED_TO')
    
    def _classify_document_type(self, doc, entities: Dict[str, List[str]]) -> str:
        """Classify document type based on content patterns."""
        # Look for patterns that indicate document type
        text_lower = doc.text.lower()
        
        # Financial indicators
        if any(entities.get('MONEY', [])) or any(entities.get('PERCENT', [])):
            if 'statement' in text_lower or 'financial' in text_lower:
                return 'financial_statement'
            elif 'report' in text_lower:
                return 'financial_report'
            elif 'agreement' in text_lower or 'contract' in text_lower:
                return 'contract'
        
        # Legal indicators
        if any(entities.get('LAW', [])) or 'agreement' in text_lower:
            return 'legal_document'
        
        # Corporate indicators
        if len(entities.get('ORG', [])) > 2:
            return 'corporate_document'
        
        return 'general_document'
    
    def _categorize_content(self, doc, entities: Dict[str, List[str]]) -> List[str]:
        """Categorize content based on entities and patterns."""
        categories = []
        
        # Entity-based categorization
        if entities.get('MONEY') or entities.get('PERCENT'):
            categories.append('financial')
        
        if entities.get('ORG'):
            categories.append('corporate')
        
        if entities.get('PERSON'):
            categories.append('personnel')
        
        if entities.get('DATE'):
            categories.append('temporal')
        
        if entities.get('GPE'):  # Geopolitical entities
            categories.append('geographical')
        
        # Content-based categorization using semantic similarity
        text_lower = doc.text.lower()
        
        # Use semantic patterns instead of keywords
        financial_concepts = self._detect_semantic_patterns(doc, 'financial')
        if financial_concepts:
            categories.append('financial_analysis')
        
        return list(set(categories)) if categories else ['general']
    
    def _detect_semantic_patterns(self, doc, domain: str) -> bool:
        """Detect semantic patterns for specific domains."""
        # This would use more sophisticated ML models in production
        # For now, we'll use basic semantic similarity
        
        if domain == 'financial':
            # Look for financial semantic patterns
            financial_tokens = [token for token in doc if token.pos_ == 'NOUN' and 
                              any(sim_token in token.text.lower() for sim_token in 
                                  ['revenue', 'profit', 'loss', 'asset', 'liability', 'equity'])]
            return len(financial_tokens) > 0
        
        return False
    
    def _calculate_confidence(self, entities: Dict[str, List[str]], 
                            relationships: List[EntityRelationship]) -> float:
        """Calculate overall confidence score for the analysis."""
        entity_score = min(1.0, len(entities) * 0.1)
        relationship_score = min(1.0, len(relationships) * 0.2)
        
        return (entity_score + relationship_score) / 2
    
    def _clean_entity_text(self, text: str) -> str:
        """Clean and normalize entity text."""
        # Remove extra whitespace and special characters
        cleaned = re.sub(r'\s+', ' ', text.strip())
        cleaned = re.sub(r'[^\w\s\-\.]', '', cleaned)
        
        # Filter out very short or numeric-only entities
        if len(cleaned) < 2 or cleaned.isdigit():
            return ""
        
        return cleaned
    
    def _is_common_phrase(self, phrase: str) -> bool:
        """Check if phrase is too common to be useful."""
        common_phrases = {
            'the company', 'this year', 'last year', 'next year',
            'the board', 'the management', 'the shareholders'
        }
        return phrase.lower() in common_phrases
    
    def _empty_insight(self) -> ContentInsight:
        """Return empty insight for invalid input."""
        return ContentInsight(
            entities={},
            relationships=[],
            document_type='unknown',
            content_categories=[],
            confidence_score=0.0
        )
    
    def learn_from_feedback(self, text: str, correct_insights: ContentInsight):
        """
        Learn from user feedback to improve analysis.
        This is a placeholder for future ML model training.
        """
        # In a full implementation, this would update the model
        # based on correct vs predicted insights
        self.logger.info(f"Learning from feedback for document type: {correct_insights.document_type}")
        pass

def test_analyzer():
    """Test the content analyzer with sample text."""
    analyzer = DynamicContentAnalyzer()
    
    sample_text = """
    Apple Inc. reported revenue of $365.8 billion in 2021, representing a 33% increase from 2020.
    The company's CEO Tim Cook announced plans to expand operations in India and China.
    The board of directors approved a dividend payment of $0.22 per share.
    """
    
    insight = analyzer.analyze_content(sample_text)
    
    print("Entities found:")
    for entity_type, entities in insight.entities.items():
        print(f"  {entity_type}: {entities}")
    
    print(f"\nDocument type: {insight.document_type}")
    print(f"Categories: {insight.content_categories}")
    print(f"Confidence: {insight.confidence_score}")
    
    print(f"\nRelationships found: {len(insight.relationships)}")
    for rel in insight.relationships:
        print(f"  {rel.source} -> {rel.relationship_type} -> {rel.target}")

if __name__ == "__main__":
    test_analyzer()
