"""
Adaptive Dgraph Manager - Dynamic graph operations without predefined schemas
Manages Dgraph operations with automatic adaptation to content structure.
"""

import logging
import json
import requests
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import asdict
from datetime import datetime
from content_analyzer import ContentInsight, EntityRelationship
from dynamic_schema_manager import DynamicSchemaManager
from enhanced_json_chunker import EnhancedChunk
from config import config

class AdaptiveDgraphManager:
    """
    Manages Dgraph operations with automatic adaptation to content.
    Creates and maintains graph structures based on ML analysis results.
    """
    
    def __init__(self, dgraph_url: str = "http://localhost:8080"):
        self.logger = logging.getLogger(__name__)
        self.dgraph_url = dgraph_url
        self.dgraph_query_url = f"{dgraph_url}/query"
        self.dgraph_mutate_url = f"{dgraph_url}/mutate"
        
        # Initialize schema manager
        self.schema_manager = DynamicSchemaManager(dgraph_url)
        
        # Track inserted nodes to avoid duplicates
        self.node_cache = {}
        
        # Initialize base schema
        self._initialize_base_schema()
    
    def _initialize_base_schema(self):
        """Initialize base schema for the graph."""
        try:
            # Define predicate schema with indexes, reverse, and count
            predicate_schema = """
            document_id: string @index(hash) .
            content: string @index(fulltext) .
            document_type: string @index(term) .
            created_at: datetime .
            updated_at: datetime .
            chunk_id: string @index(hash) .
            page_number: int .
            confidence_score: float .
            token_count: int .
            entity_id: string @index(hash) .
            entity_type: string @index(term) .
            entity_value: string @index(term) .
            confidence: float .
            # Entity relationship edges (add more as needed)
            has_person: [uid] @reverse @count .
            has_org: [uid] @reverse @count .
            has_gpe: [uid] @reverse @count .
            has_money: [uid] @reverse @count .
            has_date: [uid] @reverse @count .
            has_percent: [uid] @reverse @count .
            # Relationship edges
            rel_owns: [uid] @reverse @count .
            rel_manages: [uid] @reverse @count .
            rel_related_to: [uid] @reverse @count .
            """
            # Define types
            type_schema = """
            type Document {
                document_id: string
                content: string
                document_type: string
                created_at: datetime
                updated_at: datetime
            }
            type Chunk {
                chunk_id: string
                content: string
                document_id: string
                page_number: int
                confidence_score: float
                token_count: int
                created_at: datetime
            }
            type Entity {
                entity_id: string
                entity_type: string
                entity_value: string
                confidence: float
            }
            """
            # Apply predicate schema first, then types
            self._execute_schema_mutation(predicate_schema)
            self._execute_schema_mutation(type_schema)
            self.logger.info("Initialized base Dgraph schema with predicates and types")
        except Exception as e:
            self.logger.error(f"Error initializing base schema: {e}")
    
    def store_enhanced_chunks(self, chunks: List[EnhancedChunk]) -> bool:
        """
        Store enhanced chunks in Dgraph with their analysis results.
        
        Args:
            chunks: List of enhanced chunks to store
            
        Returns:
            Success status
        """
        if not chunks:
            return True
        
        try:
            # Update schema based on chunk analysis
            insights = [chunk.content_insight for chunk in chunks]
            self.schema_manager.analyze_and_update_schema(insights)
            
            # Store chunks and related data
            success = True
            for chunk in chunks:
                if not self._store_single_chunk(chunk):
                    success = False
            
            self.logger.info(f"Stored {len(chunks)} chunks in Dgraph")
            return success
        
        except Exception as e:
            self.logger.error(f"Error storing chunks: {e}")
            return False
    
    def _store_single_chunk(self, chunk: EnhancedChunk) -> bool:
        """Store a single enhanced chunk with all its relationships."""
        try:
            # Create main chunk node
            chunk_uid = self._create_chunk_node(chunk)
            if not chunk_uid:
                return False
            
            # Create entity nodes and relationships
            entity_uids = self._create_entity_nodes(chunk.content_insight.entities, chunk_uid)
            
            # Create relationship edges
            self._create_relationship_edges(chunk.content_insight.relationships, entity_uids)
            
            # Create document type information
            self._link_document_type(chunk_uid, chunk.content_insight.document_type)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error storing chunk {chunk.chunk_id}: {e}")
            return False
    
    def _create_chunk_node(self, chunk: EnhancedChunk) -> Optional[str]:
        """Create chunk node in Dgraph."""
        try:
            mutation = {
                "set": [{
                    "dgraph.type": "Chunk",
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "document_id": chunk.document_id,
                    "page_number": chunk.page_number,
                    "block_type": chunk.block_type,
                    "token_count": chunk.token_count,
                    "confidence_score": chunk.confidence_score,
                    "created_at": chunk.created_at
                }]
            }
            
            response = self._execute_mutation(mutation)
            if response and 'data' in response and 'uids' in response['data']:
                uid = list(response['data']['uids'].values())[0]
                self.node_cache[chunk.chunk_id] = uid
                return uid
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error creating chunk node: {e}")
            return None
    
    def _create_entity_nodes(self, entities: Dict[str, List[str]], 
                           chunk_uid: str) -> Dict[str, str]:
        """Create entity nodes and link them to chunk."""
        entity_uids = {}
        
        for entity_type, entity_values in entities.items():
            for entity_value in entity_values:
                try:
                    # Check if entity already exists
                    existing_uid = self._find_existing_entity(entity_type, entity_value)
                    
                    if existing_uid:
                        entity_uid = existing_uid
                    else:
                        # Create new entity node
                        entity_uid = self._create_new_entity(entity_type, entity_value)
                    
                    if entity_uid:
                        entity_uids[f"{entity_type}_{entity_value}"] = entity_uid
                        
                        # Link entity to chunk
                        self._link_entity_to_chunk(chunk_uid, entity_uid, entity_type)
                
                except Exception as e:
                    self.logger.error(f"Error creating entity {entity_type}:{entity_value}: {e}")
        
        return entity_uids
    
    def _find_existing_entity(self, entity_type: str, entity_value: str) -> Optional[str]:
        """Find existing entity in the graph."""
        try:
            query = f"""
            {{
                entity(func: eq(entity_value, "{entity_value}")) @filter(eq(entity_type, "{entity_type}")) {{
                    uid
                    entity_type
                    entity_value
                }}
            }}
            """
            
            response = self._execute_query(query)
            if response and 'data' in response and 'entity' in response['data']:
                entities = response['data']['entity']
                if entities:
                    return entities[0]['uid']
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error finding existing entity: {e}")
            return None
    
    def _create_new_entity(self, entity_type: str, entity_value: str) -> Optional[str]:
        """Create a new entity node."""
        try:
            entity_id = f"{entity_type}_{entity_value}_{datetime.now().timestamp()}"
            
            mutation = {
                "set": [{
                    "dgraph.type": "Entity",
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "entity_value": entity_value,
                    "confidence": 0.8,  # Default confidence
                    "created_at": datetime.now().isoformat()
                }]
            }
            
            response = self._execute_mutation(mutation)
            if response and 'data' in response and 'uids' in response['data']:
                uid = list(response['data']['uids'].values())[0]
                return uid
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error creating new entity: {e}")
            return None
    
    def _link_entity_to_chunk(self, chunk_uid: str, entity_uid: str, entity_type: str):
        """Link entity to chunk with appropriate predicate."""
        try:
            predicate_name = f"has_{entity_type.lower()}"
            
            mutation = {
                "set": [{
                    "uid": chunk_uid,
                    predicate_name: {"uid": entity_uid}
                }]
            }
            
            self._execute_mutation(mutation)
        
        except Exception as e:
            self.logger.error(f"Error linking entity to chunk: {e}")
    
    def _create_relationship_edges(self, relationships: List[EntityRelationship], 
                                 entity_uids: Dict[str, str]):
        """Create relationship edges between entities."""
        for relationship in relationships:
            try:
                source_key = self._find_entity_key(relationship.source, entity_uids)
                target_key = self._find_entity_key(relationship.target, entity_uids)
                if source_key and target_key and source_key in entity_uids and target_key in entity_uids:
                    source_uid = entity_uids[source_key]
                    target_uid = entity_uids[target_key]
                    predicate_name = f"rel_{relationship.relationship_type.lower()}"
                    facets = {"confidence": relationship.confidence, "context": relationship.context}
                    mutation = self._build_facet_mutation(source_uid, predicate_name, target_uid, facets)
                    self._execute_mutation(mutation)
            except Exception as e:
                self.logger.error(f"Error creating relationship edge: {e}")
        """Centralized helper to build a Dgraph mutation with edge facets."""
        mutation_obj = {
            "uid": source_uid,
            predicate_name: {"uid": target_uid}
        }
        for facet_key, facet_value in facets.items():
            mutation_obj[f"{predicate_name}|{facet_key}"] = facet_value
        return {"set": [mutation_obj]}

        """Ensure no object mixes node fields and edge facets for the same predicate."""
        for obj in mutation.get("set", []):
            for key in obj:
                if "|" in key:
                    pred, facet = key.split("|", 1)
                    if pred not in obj or not (isinstance(obj[pred], dict) and "uid" in obj[pred]):
                        self.logger.error(f"Facet {key} present but base edge {pred} missing or not a uid edge.")
                        return False
        return True
    
    def _find_entity_key(self, entity_value: str, entity_uids: Dict[str, str]) -> Optional[str]:
        """Find the key for an entity value in the entity_uids dict."""
        for key in entity_uids:
            if entity_value in key:
                return key
        return None
    
    def _link_document_type(self, chunk_uid: str, document_type: str):
        """Link chunk to its document type."""
        try:
            mutation = {
                "set": [{
                    "uid": chunk_uid,
                    "document_type": document_type
                }]
            }
            
            self._execute_mutation(mutation)
        
        except Exception as e:
            self.logger.error(f"Error linking document type: {e}")
    
    def query_related_chunks(self, entity_value: str, entity_type: str = None) -> List[Dict]:
        """Query chunks related to a specific entity using robust reverse edge logic."""
        try:
            # Known has_* predicates (update as needed for your schema)
            has_predicates = [
                'has_person', 'has_org', 'has_gpe', 'has_money', 'has_date', 'has_percent'
            ]
            if entity_type and f"has_{entity_type.lower()}" in has_predicates:
                # Use specific predicate if valid
                predicate_name = f"has_{entity_type.lower()}"
                entity_filter = f'@filter(eq(entity_type, "{entity_type}"))'
                query = f"""
                {{
                    entity(func: eq(entity_value, "{entity_value}")) {entity_filter} {{
                        uid
                        entity_value
                        entity_type
                        ~{predicate_name} {{
                            uid
                            chunk_id
                            content
                            confidence_score
                            document_id
                            page_number
                        }}
                    }}
                }}
                """
                response = self._execute_query(query)
                if response and 'data' in response and 'entity' in response['data']:
                    entities = response['data']['entity']
                    chunks = []
                    for entity in entities:
                        if f"~{predicate_name}" in entity:
                            chunks.extend(entity[f"~{predicate_name}"])
                    return chunks
                return []
            else:
                # No or unknown entity_type: query all has_* edges for the found entity UID(s)
                # Build a query that unions all ~has_* edges
                edge_blocks = '\n'.join([
                    f"~{pred} {{\n    uid\n    chunk_id\n    content\n    confidence_score\n    document_id\n    page_number\n}}" for pred in has_predicates
                ])
                query = f"""
                {{
                    entity(func: eq(entity_value, "{entity_value}")) {{
                        uid
                        entity_value
                        entity_type
                        {edge_blocks}
                    }}
                }}
                """
                response = self._execute_query(query)
                if response and 'data' in response and 'entity' in response['data']:
                    entities = response['data']['entity']
                    chunks = []
                    for entity in entities:
                        for pred in has_predicates:
                            edge = f"~{pred}"
                            if edge in entity:
                                chunks.extend(entity[edge])
                    return chunks
                return []
        except Exception as e:
            self.logger.error(f"Error querying related chunks: {e}")
            return []
    
    def query_entity_relationships(self, entity_value: str) -> List[Dict]:
        """Query all relationships for a specific entity."""
        try:
            query = f"""
            {{
                entity(func: eq(entity_value, "{entity_value}")) {{
                    uid
                    entity_value
                    entity_type
                    ~rel_owns {{
                        entity_value
                        entity_type
                    }}
                    ~rel_manages {{
                        entity_value
                        entity_type
                    }}
                    ~rel_related_to {{
                        entity_value
                        entity_type
                    }}
                }}
            }}
            """
            
            response = self._execute_query(query)
            if response and 'data' in response:
                return response['data'].get('entity', [])
            
            return []
        
        except Exception as e:
            self.logger.error(f"Error querying entity relationships: {e}")
            return []

    def query_entity_occurrences(self, entity_value: str) -> Dict[str, Any]:
        """
        Comprehensive query to find all occurrences of an entity across all entity types.
        Uses proper reverse edges to find all chunks mentioning this entity.
        """
        try:
            query = f"""
            {{
                entity(func: eq(entity_value, "{entity_value}")) {{
                    uid
                    entity_value
                    entity_type
                    # Find chunks through reverse edges for all possible entity types
                    ~has_person {{
                        uid
                        chunk_id
                        content
                        document_id
                        confidence_score
                    }}
                    ~has_org {{
                        uid
                        chunk_id
                        content
                        document_id
                        confidence_score
                    }}
                    ~has_gpe {{
                        uid
                        chunk_id
                        content
                        document_id
                        confidence_score
                    }}
                    ~has_money {{
                        uid
                        chunk_id
                        content
                        document_id
                        confidence_score
                    }}
                    ~has_date {{
                        uid
                        chunk_id
                        content
                        document_id
                        confidence_score
                    }}
                    ~has_percent {{
                        uid
                        chunk_id
                        content
                        document_id
                        confidence_score
                    }}
                    # Entity relationships
                    rel_owns {{
                        uid
                        entity_value
                        entity_type
                    }}
                    rel_manages {{
                        uid
                        entity_value
                        entity_type
                    }}
                    rel_related_to {{
                        uid
                        entity_value
                        entity_type
                    }}
                }}
            }}
            """
            
            response = self._execute_query(query)
            if response and 'data' in response and 'entity' in response['data']:
                entities = response['data']['entity']
                
                # Consolidate results
                result = {
                    'entity_info': [],
                    'related_chunks': [],
                    'relationships': [],
                    'total_occurrences': 0
                }
                
                for entity in entities:
                    result['entity_info'].append({
                        'uid': entity.get('uid'),
                        'value': entity.get('entity_value'),
                        'type': entity.get('entity_type')
                    })
                    
                    # Collect chunks from all reverse edge types
                    for edge_type in ['~has_person', '~has_org', '~has_gpe', '~has_money', '~has_date', '~has_percent']:
                        if edge_type in entity:
                            result['related_chunks'].extend(entity[edge_type])
                    
                    # Collect relationships
                    for rel_type in ['rel_owns', 'rel_manages', 'rel_related_to']:
                        if rel_type in entity:
                            result['relationships'].extend(entity[rel_type])
                
                result['total_occurrences'] = len(result['related_chunks'])
                return result
            
            return {'entity_info': [], 'related_chunks': [], 'relationships': [], 'total_occurrences': 0}
        
        except Exception as e:
            self.logger.error(f"Error querying entity occurrences: {e}")
            return {'entity_info': [], 'related_chunks': [], 'relationships': [], 'total_occurrences': 0}
    
    def get_document_graph(self, document_id: str) -> Dict[str, Any]:
        """Get complete graph structure for a document."""
        try:
            query = f"""
            {{
                document(func: eq(document_id, "{document_id}")) {{
                    uid
                    chunk_id
                    content
                    page_number
                    confidence_score
                    has_org {{
                        uid
                        entity_value
                        entity_type
                        rel_owns {{
                            entity_value
                            entity_type
                        }}
                        rel_manages {{
                            entity_value
                            entity_type
                        }}
                    }}
                    has_person {{
                        uid
                        entity_value
                        entity_type
                        rel_related_to {{
                            entity_value
                            entity_type
                        }}
                    }}
                    has_money {{
                        uid
                        entity_value
                        entity_type
                    }}
                    has_date {{
                        uid
                        entity_value
                        entity_type
                    }}
                }}
            }}
            """
            
            response = self._execute_query(query)
            if response and 'data' in response:
                return response['data']
            
            return {}
        
        except Exception as e:
            self.logger.error(f"Error getting document graph: {e}")
            return {}
    
    def _execute_query(self, query: str) -> Optional[Dict]:
        """Execute a GraphQL query on Dgraph."""
        try:
            response = requests.post(
                self.dgraph_query_url,
                json={"query": query}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Query failed: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return None
    
    def _execute_mutation(self, mutation: Dict) -> Optional[Dict]:
        """Execute a mutation on Dgraph."""
        try:
            response = requests.post(
                self.dgraph_mutate_url,
                json=mutation
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Mutation failed: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            self.logger.error(f"Error executing mutation: {e}")
            return None
    
    def _execute_schema_mutation(self, schema: str) -> bool:
        """Execute schema mutation on Dgraph."""
        try:
            response = requests.post(
                f"{self.dgraph_url}/alter",
                data=schema,
                headers={'Content-Type': 'text/plain'}
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Error executing schema mutation: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        try:
            query = """
            {
                chunk_count(func: type(Chunk)) {
                    count(uid)
                }
                entity_count(func: type(Entity)) {
                    count(uid)
                }
                document_count(func: type(Document)) {
                    count(uid)
                }
            }
            """
            
            response = self._execute_query(query)
            if response and 'data' in response:
                return {
                    'chunks': response['data'].get('chunk_count', [{}])[0].get('count', 0),
                    'entities': response['data'].get('entity_count', [{}])[0].get('count', 0),
                    'documents': response['data'].get('document_count', [{}])[0].get('count', 0)
                }
            
            return {}
        
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    def cleanup_graph(self) -> bool:
        """Clean up the entire graph (use with caution)."""
        try:
            # Drop all data
            mutation = {"drop_all": True}
            response = self._execute_mutation(mutation)
            
            if response:
                # Reinitialize base schema
                self._initialize_base_schema()
                self.node_cache.clear()
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error cleaning up graph: {e}")
            return False

def test_adaptive_dgraph():
    """Test the adaptive Dgraph manager."""
    from enhanced_json_chunker import EnhancedJSONChunker
    
    # Create sample chunks
    chunker = EnhancedJSONChunker()
    
    # Sample data
    sample_data = [{
        'document_id': 'test_doc_001',
        'pages': [{
            'blocks': [{
                'type': 'text',
                'content': 'Apple Inc. reported revenue of $365.8 billion in 2021. Tim Cook is the CEO.'
            }]
        }]
    }]
    
    # Save and process sample data
    with open('test_extraction.json', 'w') as f:
        json.dump(sample_data, f)
    
    chunks = chunker.process_extracted_json('test_extraction.json')
    
    # Test Dgraph operations
    dgraph_manager = AdaptiveDgraphManager()
    
    # Store chunks
    success = dgraph_manager.store_enhanced_chunks(chunks)
    print(f"Stored chunks successfully: {success}")
    
    # Query related chunks
    related = dgraph_manager.query_related_chunks("Apple Inc.", "ORG")
    print(f"Found {len(related)} related chunks")
    
    # Get statistics
    stats = dgraph_manager.get_statistics()
    print(f"Graph statistics: {stats}")

if __name__ == "__main__":
    test_adaptive_dgraph()
