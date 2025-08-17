"""
Dynamic Schema Manager for Dgraph
Automatically manages graph schema evolution based on content analysis.
"""

import logging
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, asdict
import json
import requests
from datetime import datetime
from content_analyzer import ContentInsight, EntityRelationship
from config import config

@dataclass
class SchemaUpdate:
    """Represents a schema update operation."""
    operation: str  # 'add_type', 'add_predicate', 'modify_predicate'
    element_type: str  # 'node', 'edge'
    name: str
    properties: Dict[str, Any]
    timestamp: datetime

class DynamicSchemaManager:
    """
    Manages Dgraph schema dynamically based on content analysis.
    Automatically evolves schema without predefined structures.
    """
    
    def __init__(self, dgraph_url: str = "http://localhost:8080"):
        self.logger = logging.getLogger(__name__)
        self.dgraph_url = dgraph_url
        self.dgraph_admin_url = f"{dgraph_url}/admin"
        self.current_schema = {}
        self.schema_updates = []
        
        # Base schema elements that are always present
        self.base_predicates = {
            'dgraph.type': 'string',
            'content': 'string',
            'confidence': 'float',
            'document_id': 'string',
            'chunk_id': 'string',
            'created_at': 'datetime',
            'updated_at': 'datetime'
        }
        
        # Load current schema
        self._load_current_schema()
    
    def _load_current_schema(self):
        """Load current schema from Dgraph."""
        try:
            response = requests.get(f"{self.dgraph_admin_url}/schema")
            if response.status_code == 200:
                schema_data = response.json()
                self.current_schema = self._parse_schema_response(schema_data)
                self.logger.info(f"Loaded current schema with {len(self.current_schema)} predicates")
            else:
                self.logger.warning(f"Could not load schema: {response.status_code}")
                self.current_schema = {}
        except Exception as e:
            self.logger.error(f"Error loading schema: {e}")
            self.current_schema = {}
    
    def _parse_schema_response(self, schema_data: Dict) -> Dict[str, str]:
        """Parse Dgraph schema response into predicate -> type mapping."""
        predicates = {}
        
        if 'schema' in schema_data:
            for predicate in schema_data['schema']:
                if 'predicate' in predicate and 'type' in predicate:
                    predicates[predicate['predicate']] = predicate['type']
        
        return predicates
    
    def analyze_and_update_schema(self, insights: List[ContentInsight]) -> List[SchemaUpdate]:
        """
        Analyze content insights and update schema as needed.
        
        Args:
            insights: List of content insights from content analyzer
            
        Returns:
            List of schema updates that were applied
        """
        updates = []
        # --- Add dedicated Entity node schema if not present ---
        # Ensure entity_type and entity_value predicates are indexed for fast lookup
        for pred, tokenizer in [("entity_type", ["hash", "term"]), ("entity_value", ["hash", "term"])]:
            if pred not in self.current_schema:
                updates.append(SchemaUpdate(
                    operation='add_predicate',
                    element_type='node',
                    name=pred,
                    properties={
                        'type': 'string',
                        'index': True,
                        'tokenizer': tokenizer
                    },
                    timestamp=datetime.now()
                ))
        # --- End Entity node schema ---
        for insight in insights:
            # Analyze entities and create predicates
            entity_updates = self._analyze_entities_for_schema(insight.entities)
            updates.extend(entity_updates)

            # Analyze relationships and create edge predicates
            relationship_updates = self._analyze_relationships_for_schema(insight.relationships)
            updates.extend(relationship_updates)

            # Create document type predicates (with strong indexing)
            doc_type_updates = self._analyze_document_type_for_schema(insight.document_type)
            updates.extend(doc_type_updates)

        # Apply updates to Dgraph
        applied_updates = self._apply_schema_updates(updates)
        return applied_updates
    
    def _analyze_entities_for_schema(self, entities: Dict[str, List[str]]) -> List[SchemaUpdate]:
        """Analyze entities and determine required schema updates."""
        updates = []
        for entity_type, entity_list in entities.items():
            if not entity_list:
                continue
            # Create predicate for this entity type (edge)
            predicate_name = f"has_{entity_type.lower()}"
            if predicate_name not in self.current_schema:
                update = SchemaUpdate(
                    operation='add_predicate',
                    element_type='edge',
                    name=predicate_name,
                    properties={
                        'type': '[uid]',
                        'reverse': True,
                        'count': True
                    },
                    timestamp=datetime.now()
                )
                updates.append(update)
        return updates
    
    def _analyze_relationships_for_schema(self, relationships: List[EntityRelationship]) -> List[SchemaUpdate]:
        """Analyze relationships and determine required schema updates."""
        updates = []
        
        # Group relationships by type
        relationship_types = set()
        for rel in relationships:
            relationship_types.add(rel.relationship_type)
        
        for rel_type in relationship_types:
            predicate_name = f"rel_{rel_type.lower()}"
            
            if predicate_name not in self.current_schema:
                update = SchemaUpdate(
                    operation='add_predicate',
                    element_type='edge',
                    name=predicate_name,
                    properties={
                        'type': '[uid]',  # Reference to other nodes
                        'reverse': True,  # Enable reverse edges
                        'count': True     # Enable count index
                    },
                    timestamp=datetime.now()
                )
                updates.append(update)
        
        return updates
    
    def _analyze_document_type_for_schema(self, document_type: str) -> List[SchemaUpdate]:
        """Analyze document type and create corresponding schema elements."""
        updates = []
        # Create type predicate for document classification
        type_predicate = f"document_type"
        if type_predicate not in self.current_schema:
            update = SchemaUpdate(
                operation='add_predicate',
                element_type='node',
                name=type_predicate,
                properties={
                    'type': 'string',
                    'index': True,
                    'tokenizer': ['exact', 'hash', 'term']
                },
                timestamp=datetime.now()
            )
            updates.append(update)
        return updates
    
    def _apply_schema_updates(self, updates: List[SchemaUpdate]) -> List[SchemaUpdate]:
        """Apply schema updates to Dgraph."""
        applied_updates = []
        
        if not updates:
            return applied_updates
        
        # Group updates by operation type
        schema_changes = []
        
        for update in updates:
            if update.operation == 'add_predicate':
                schema_def = self._create_predicate_definition(update)
                schema_changes.append(schema_def)
        
        if schema_changes:
            try:
                # Apply schema changes
                schema_mutation = '\n'.join(schema_changes)
                response = self._execute_schema_mutation(schema_mutation)
                
                if response:
                    applied_updates = updates
                    self.schema_updates.extend(updates)
                    
                    # Update current schema cache
                    for update in updates:
                        self.current_schema[update.name] = update.properties.get('type', 'string')
                    
                    self.logger.info(f"Applied {len(updates)} schema updates successfully")
                else:
                    self.logger.error("Failed to apply schema updates")
            
            except Exception as e:
                self.logger.error(f"Error applying schema updates: {e}")
        
        return applied_updates
    
    def _create_predicate_definition(self, update: SchemaUpdate) -> str:
        """Create Dgraph predicate definition from schema update."""
        properties = update.properties
        predicate_type = properties.get('type', 'string')
        
        definition_parts = [f"{update.name}: {predicate_type}"]
        
        # Add indices if specified
        if properties.get('index', False):
            tokenizers = properties.get('tokenizer', ['term'])
            tokenizer_str = ','.join(tokenizers)
            definition_parts.append(f"@index({tokenizer_str})")
        
        # Add reverse if specified
        if properties.get('reverse', False):
            definition_parts.append("@reverse")
        
        # Add count if specified
        if properties.get('count', False):
            definition_parts.append("@count")
        
        return ' '.join(definition_parts) + ' .'
    
    def _execute_schema_mutation(self, schema_mutation: str) -> bool:
        """Execute schema mutation on Dgraph."""
        try:
            headers = {'Content-Type': 'text/plain'}
            response = requests.post(
                f"{self.dgraph_admin_url}/schema",
                data=schema_mutation,
                headers=headers
            )
            
            if response.status_code == 200:
                self.logger.info("Schema mutation executed successfully")
                return True
            else:
                self.logger.error(f"Schema mutation failed: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error executing schema mutation: {e}")
            return False
    
    def get_schema_evolution_history(self) -> List[Dict]:
        """Get history of schema evolution."""
        return [asdict(update) for update in self.schema_updates]
    
    def create_node_type(self, type_name: str, properties: Dict[str, str]) -> bool:
        """Create a new node type with specified properties."""
        updates = []
        
        # Create predicates for each property
        for prop_name, prop_type in properties.items():
            predicate_name = f"{type_name.lower()}_{prop_name}"
            
            update = SchemaUpdate(
                operation='add_predicate',
                element_type='node',
                name=predicate_name,
                properties={
                    'type': prop_type,
                    'index': True,
                    'tokenizer': ['term'] if prop_type == 'string' else []
                },
                timestamp=datetime.now()
            )
            updates.append(update)
        
        applied = self._apply_schema_updates(updates)
        return len(applied) == len(updates)
    
    def suggest_schema_optimizations(self) -> List[str]:
        """Suggest optimizations based on usage patterns."""
        suggestions = []
        
        # Analyze current schema usage
        predicate_usage = self._analyze_predicate_usage()
        
        # Suggest indices for frequently queried predicates
        for predicate, usage_count in predicate_usage.items():
            if usage_count > 100 and predicate not in self.current_schema:
                suggestions.append(f"Consider adding index to predicate: {predicate}")
        
        # Suggest reverse edges for relationship predicates
        for predicate in self.current_schema:
            if predicate.startswith('rel_') and '@reverse' not in str(self.current_schema[predicate]):
                suggestions.append(f"Consider adding reverse edge to: {predicate}")
        
        return suggestions
    
    def _analyze_predicate_usage(self) -> Dict[str, int]:
        """Analyze predicate usage patterns."""
        # This would query Dgraph for actual usage statistics
        # For now, return mock data
        return {
            'has_org': 150,
            'has_person': 200,
            'rel_owns': 75,
            'document_type': 300
        }
    
    def export_schema(self) -> Dict[str, Any]:
        """Export current schema for backup or migration."""
        return {
            'predicates': self.current_schema,
            'updates_history': self.get_schema_evolution_history(),
            'exported_at': datetime.now().isoformat()
        }
    
    def import_schema(self, schema_data: Dict[str, Any]) -> bool:
        """Import schema from backup."""
        try:
            predicates = schema_data.get('predicates', {})
            
            # Create updates for missing predicates
            updates = []
            for pred_name, pred_type in predicates.items():
                if pred_name not in self.current_schema:
                    update = SchemaUpdate(
                        operation='add_predicate',
                        element_type='node',
                        name=pred_name,
                        properties={'type': pred_type},
                        timestamp=datetime.now()
                    )
                    updates.append(update)
            
            applied = self._apply_schema_updates(updates)
            return len(applied) == len(updates)
        
        except Exception as e:
            self.logger.error(f"Error importing schema: {e}")
            return False

def test_schema_manager():
    """Test the schema manager with sample insights."""
    from content_analyzer import DynamicContentAnalyzer, EntityRelationship
    
    # Create sample insights
    sample_insight = DynamicContentAnalyzer().analyze_content(
        "Apple Inc. reported revenue of $365.8 billion in 2021."
    )
    
    # Initialize schema manager
    schema_manager = DynamicSchemaManager()
    
    # Analyze and update schema
    updates = schema_manager.analyze_and_update_schema([sample_insight])
    
    print(f"Applied {len(updates)} schema updates:")
    for update in updates:
        print(f"  - {update.operation}: {update.name}")
    
    # Export schema
    schema_export = schema_manager.export_schema()
    print(f"\nCurrent schema has {len(schema_export['predicates'])} predicates")

if __name__ == "__main__":
    test_schema_manager()
