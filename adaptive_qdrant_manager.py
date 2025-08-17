"""
Adaptive Qdrant Manager - Dynamic vector operations with flexible metadata
Manages Qdrant vector database operations with automatic adaptation to content structure.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from dataclasses import asdict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from enhanced_json_chunker import EnhancedChunk
from content_analyzer import ContentInsight
from config import config

class AdaptiveQdrantManager:
    """
    Manages Qdrant vector database operations with automatic adaptation to content.
    Creates dynamic collections based on content analysis results.
    """
    
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "adaptive_chunks"):
        self.logger = logging.getLogger(__name__)
        self.qdrant_url = qdrant_url
        self.base_collection_name = collection_name
        
        # Initialize Qdrant client
        try:
            self.client = QdrantClient(url=qdrant_url)
            self.logger.info(f"Connected to Qdrant at {qdrant_url}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            raise
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
            self.logger.info("Initialized embedding model")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            raise
        
        # Track dynamic collections
        self.collections = set()
        
        # Initialize base collection
        self._initialize_base_collection()
    
    def _initialize_base_collection(self):
        """Initialize the base collection for all chunks."""
        try:
            collection_name = self.base_collection_name
            
            # Check if collection exists
            existing_collections = self.client.get_collections()
            collection_exists = any(col.name == collection_name for col in existing_collections.collections)
            
            if not collection_exists:
                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Created base collection: {collection_name}")
            
            self.collections.add(collection_name)
        
        except Exception as e:
            self.logger.error(f"Error initializing base collection: {e}")
    
    def store_enhanced_chunks(self, chunks: List[EnhancedChunk], 
                            collection_name: Optional[str] = None) -> bool:
        """
        Store enhanced chunks in Qdrant with adaptive metadata.
        
        Args:
            chunks: List of enhanced chunks to store
            collection_name: Optional specific collection name
            
        Returns:
            Success status
        """
        if not chunks:
            return True
        
        try:
            target_collection = collection_name or self.base_collection_name
            
            # Ensure collection exists
            self._ensure_collection_exists(target_collection)
            
            # Prepare points for insertion
            points = []
            for chunk in chunks:
                point = self._create_point_from_chunk(chunk)
                if point:
                    points.append(point)
            
            if points:
                # Insert points in batches
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    
                    operation_result = self.client.upsert(
                        collection_name=target_collection,
                        points=batch
                    )
                    
                    if not operation_result:
                        self.logger.error(f"Failed to insert batch {i//batch_size + 1}")
                        return False
                
                self.logger.info(f"Stored {len(points)} chunks in collection: {target_collection}")
                return True
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error storing chunks: {e}")
            return False
    
    def _create_point_from_chunk(self, chunk: EnhancedChunk) -> Optional[PointStruct]:
        """Create a Qdrant point from an enhanced chunk."""
        try:
            # Use existing embedding or create new one
            if chunk.semantic_embedding:
                vector = chunk.semantic_embedding
            else:
                vector = self.embedding_model.encode(chunk.content).tolist()
            
            # Create adaptive metadata
            metadata = self._create_adaptive_metadata(chunk)
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=metadata
            )
            
            return point
        
        except Exception as e:
            self.logger.error(f"Error creating point from chunk: {e}")
            return None
    
    def _create_adaptive_metadata(self, chunk: EnhancedChunk) -> Dict[str, Any]:
        """Create adaptive metadata based on chunk content analysis."""
        # Store both content and snippet for Qdrant payload
        snippet = chunk.content[:200] + ("..." if len(chunk.content) > 200 else "")
        metadata = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_type": chunk.content_insight.document_type,
            "snippet": snippet,
            "content": chunk.content,
            "page_number": chunk.page_number,
            "block_type": chunk.block_type,
            "entities": chunk.content_insight.entities
        }
        # Add dynamic has_{ENTITY_TYPE} booleans for all detected entity types
        for entity_type in chunk.content_insight.entities:
            key = f"has_{entity_type.upper()}"
            metadata[key] = bool(chunk.content_insight.entities.get(entity_type))
        return metadata
    
    def _ensure_collection_exists(self, collection_name: str):
        """Ensure a collection exists, create if it doesn't."""
        try:
            if collection_name not in self.collections:
                existing_collections = self.client.get_collections()
                collection_exists = any(col.name == collection_name for col in existing_collections.collections)
                
                if not collection_exists:
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=self.embedding_dimension,
                            distance=Distance.COSINE
                        )
                    )
                    self.logger.info(f"Created collection: {collection_name}")
                
                self.collections.add(collection_name)
        
        except Exception as e:
            self.logger.error(f"Error ensuring collection exists: {e}")
    
    def semantic_search(self, query_text: str, 
                       collection_name: Optional[str] = None,
                       top_k: int = 10,
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search with optional filters.
        
        Args:
            query_text: Text to search for
            collection_name: Collection to search in
            top_k: Number of results to return
            filters: Optional filters to apply
            
        Returns:
            List of search results with metadata
        """
        try:
            target_collection = collection_name or self.base_collection_name
            
            # Generate query embedding
            query_vector = self.embedding_model.encode(query_text).tolist()
            
            # Build filter conditions
            filter_conditions = self._build_filter_conditions(filters) if filters else None
            
            # Perform search
            search_results = self.client.search(
                collection_name=target_collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter_conditions
            )
            
            # Format results
            results = []
            for result in search_results:
                formatted_result = {
                    "id": result.id,
                    "score": result.score,
                    "content": result.payload.get("content", ""),
                    "chunk_id": result.payload.get("chunk_id", ""),
                    "document_id": result.payload.get("document_id", ""),
                    "document_type": result.payload.get("document_type", ""),
                    "confidence_score": result.payload.get("confidence_score", 0.0),
                    "entities": self._extract_entities_from_payload(result.payload),
                    "relationships": self._extract_relationships_from_payload(result.payload),
                    "metadata": result.payload
                }
                results.append(formatted_result)
            
            self.logger.info(f"Found {len(results)} results for query: {query_text[:50]}...")
            return results
        
        except Exception as e:
            self.logger.error(f"Error performing semantic search: {e}")
            return []
    
    def _build_filter_conditions(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """Build Qdrant filter conditions from filter dictionary."""
        try:
            from qdrant_client.http.models import MatchAny, MatchValue, FieldCondition, Filter
            conditions = []
            for key, value in filters.items():
                if isinstance(value, str):
                    condition = FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                elif isinstance(value, list):
                    # Use MatchAny for lists
                    condition = FieldCondition(
                        key=key,
                        match=MatchAny(any=value)
                    )
                elif isinstance(value, bool):
                    condition = FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                else:
                    continue  # Skip unsupported types
                conditions.append(condition)
            if conditions:
                return Filter(must=conditions)
            return None
        except Exception as e:
            self.logger.error(f"Error building filter conditions: {e}")
            return None
    
    def _extract_entities_from_payload(self, payload: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract entities from payload metadata."""
        entities = {}
        
        for key, value in payload.items():
            if key.startswith("entities_") and isinstance(value, list):
                entity_type = key.replace("entities_", "").upper()
                entities[entity_type] = value
        
        return entities
    
    def _extract_relationships_from_payload(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relationships from payload metadata."""
        relationships = []
        
        # Find relationship entries
        rel_indices = set()
        for key in payload.keys():
            if key.startswith("rel_") and key.endswith("_type"):
                index = key.split("_")[1]
                rel_indices.add(int(index))
        
        # Reconstruct relationships
        for index in sorted(rel_indices):
            rel = {
                "type": payload.get(f"rel_{index}_type", ""),
                "source": payload.get(f"rel_{index}_source", ""),
                "target": payload.get(f"rel_{index}_target", ""),
                "confidence": payload.get(f"rel_{index}_confidence", 0.0)
            }
            relationships.append(rel)
        
        return relationships
    
    def filter_by_document_type(self, document_type: str, 
                               collection_name: Optional[str] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Filter chunks by document type."""
        filters = {"document_type": document_type}
        try:
            target_collection = collection_name or self.base_collection_name
            filter_conditions = self._build_filter_conditions(filters)
            # Use scroll for filter-only queries (no vector)
            scroll_result = self.client.scroll(
                collection_name=target_collection,
                scroll_filter=filter_conditions,
                limit=limit
            )
            # scroll returns (points, next_page_offset)
            points = scroll_result[0] if isinstance(scroll_result, (list, tuple)) else scroll_result
            return [{"id": p.id, "payload": p.payload} for p in points]
        except Exception as e:
            self.logger.error(f"Error filtering by document type: {e}")
            return []
    
    def find_chunks_with_entities(self, entity_type: str, entity_value: Optional[str] = None,
                                 collection_name: Optional[str] = None,
                                 limit: int = 50) -> List[Dict[str, Any]]:
        """Find chunks containing specific entities."""
        try:
            target_collection = collection_name or self.base_collection_name
            
            # Build filter for entity presence
            filters = {f"has_{entity_type.lower()}": True}
            
            # Add specific entity value if provided
            if entity_value:
                # This would require a more sophisticated approach in production
                # For now, we'll search semantically for the entity value
                query_vector = self.embedding_model.encode(entity_value).tolist()
            else:
                query_vector = [0.0] * self.embedding_dimension
            
            filter_conditions = self._build_filter_conditions(filters)
            
            results = self.client.search(
                collection_name=target_collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter_conditions
            )
            
            return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]
        
        except Exception as e:
            self.logger.error(f"Error finding chunks with entities: {e}")
            return []
    
    def get_similar_chunks(self, chunk_id: str, top_k: int = 5,
                          collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find chunks similar to a given chunk."""
        try:
            target_collection = collection_name or self.base_collection_name
            
            # First, get the chunk vector
            chunk_results = self.client.scroll(
                collection_name=target_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="chunk_id",
                            match=MatchValue(value=chunk_id)
                        )
                    ]
                ),
                limit=1,
                with_vectors=True
            )
            
            if not chunk_results[0]:
                return []
            
            chunk_vector = chunk_results[0][0].vector
            
            # Find similar chunks
            similar_results = self.client.search(
                collection_name=target_collection,
                query_vector=chunk_vector,
                limit=top_k + 1  # +1 to exclude the original chunk
            )
            
            # Filter out the original chunk
            filtered_results = [r for r in similar_results if r.payload.get("chunk_id") != chunk_id]
            
            return [{"id": r.id, "score": r.score, "payload": r.payload} for r in filtered_results[:top_k]]
        
        except Exception as e:
            self.logger.error(f"Error finding similar chunks: {e}")
            return []
    
    def get_collection_statistics(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about a collection."""
        try:
            target_collection = collection_name or self.base_collection_name
            
            collection_info = self.client.get_collection(target_collection)
            
            # Get some sample points to analyze metadata patterns
            sample_results = self.client.scroll(
                collection_name=target_collection,
                limit=100,
                with_payload=True
            )
            
            # Analyze metadata patterns
            metadata_fields = set()
            document_types = set()
            entity_types = set()
            
            for point in sample_results[0]:
                metadata_fields.update(point.payload.keys())
                
                if "document_type" in point.payload:
                    document_types.add(point.payload["document_type"])
                
                for key in point.payload.keys():
                    if key.startswith("has_") and point.payload[key]:
                        entity_type = key.replace("has_", "").upper()
                        entity_types.add(entity_type)
            
            return {
                "collection_name": target_collection,
                "total_points": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value,
                "metadata_fields": list(metadata_fields),
                "document_types": list(document_types),
                "entity_types": list(entity_types)
            }
        
        except Exception as e:
            self.logger.error(f"Error getting collection statistics: {e}")
            return {}
    
    def create_specialized_collection(self, collection_name: str, document_type: str) -> bool:
        """Create a specialized collection for a specific document type."""
        try:
            # Create new collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            
            self.collections.add(collection_name)
            self.logger.info(f"Created specialized collection: {collection_name} for {document_type}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error creating specialized collection: {e}")
            return False
    
    def migrate_chunks_by_type(self, document_type: str, target_collection: str) -> bool:
        """Migrate chunks of a specific document type to a specialized collection."""
        try:
            # Ensure target collection exists
            self._ensure_collection_exists(target_collection)
            
            # Find all chunks of the specified document type
            chunks_to_migrate = self.filter_by_document_type(document_type, limit=1000)
            
            if not chunks_to_migrate:
                return True
            
            # Get full point data
            point_ids = [chunk["id"] for chunk in chunks_to_migrate]
            
            source_points = self.client.retrieve(
                collection_name=self.base_collection_name,
                ids=point_ids,
                with_payload=True,
                with_vectors=True
            )
            
            # Insert into target collection
            operation_result = self.client.upsert(
                collection_name=target_collection,
                points=source_points
            )
            
            if operation_result:
                # Remove from source collection
                self.client.delete(
                    collection_name=self.base_collection_name,
                    points_selector=models.PointIdsList(points=point_ids)
                )
                
                self.logger.info(f"Migrated {len(chunks_to_migrate)} chunks to {target_collection}")
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error migrating chunks: {e}")
            return False
    
    def cleanup_collection(self, collection_name: Optional[str] = None) -> bool:
        """Clean up a collection (delete all points)."""
        try:
            target_collection = collection_name or self.base_collection_name
            
            # Delete the entire collection
            self.client.delete_collection(target_collection)
            
            # Remove from tracking
            if target_collection in self.collections:
                self.collections.remove(target_collection)
            
            # Recreate if it was the base collection
            if target_collection == self.base_collection_name:
                self._initialize_base_collection()
            
            self.logger.info(f"Cleaned up collection: {target_collection}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error cleaning up collection: {e}")
            return False

def test_adaptive_qdrant():
    """Test the adaptive Qdrant manager."""
    from enhanced_json_chunker import EnhancedJSONChunker
    
    # Create sample chunks
    chunker = EnhancedJSONChunker()
    
    sample_data = [{
        'document_id': 'test_doc_001',
        'pages': [{
            'blocks': [{
                'type': 'text',
                'content': 'Apple Inc. reported revenue of $365.8 billion in 2021. Tim Cook is the CEO.'
            }]
        }]
    }]
    
    # Save and process
    with open('test_extraction.json', 'w') as f:
        json.dump(sample_data, f)
    
    chunks = chunker.process_extracted_json('test_extraction.json')
    
    # Test Qdrant operations
    qdrant_manager = AdaptiveQdrantManager()
    
    # Store chunks
    success = qdrant_manager.store_enhanced_chunks(chunks)
    print(f"Stored chunks successfully: {success}")
    
    # Semantic search
    results = qdrant_manager.semantic_search("Apple financial results", top_k=5)
    print(f"Found {len(results)} search results")
    for result in results:
        print(f"  Score: {result['score']:.3f} - {result['content'][:100]}...")
    
    # Get statistics
    stats = qdrant_manager.get_collection_statistics()
    print(f"Collection statistics: {stats}")

if __name__ == "__main__":
    test_adaptive_qdrant()
