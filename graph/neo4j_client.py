from typing import List, Dict, Any
from neo4j import GraphDatabase, Driver
import os

class Neo4jGraphClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver : Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def add_document_kg(self, entities: List[Dict[str, str]], relations: List[Dict[str, str]]) -> None:
        """
        Idempotently inserts extracted entities and relations into Neo4j.
        entities: [{'name': 'AuthService', 'type': 'Service'}, ...]
        relations: [{'source': 'AuthService', 'target': 'UserDB', 'type': 'DEPENDS_ON'}, ...]
        """
        with self.driver.session() as session:
            # 1. Merge Entities (Nodes)
            for entity in entities:
                query = (
                    "MERGE (e:Entity {name: $name}) "
                    "ON CREATE SET e.type = $type"
                )
                session.run(query, name=entity['name'], type=entity['type'])

            # 2. Merge Relations (Edges)
            for rel in relations:
                query = (
                    "MATCH (s:Entity {name: $source}) "
                    "MATCH (t:Entity {name: $target}) "
                    "MERGE (s)-[r:RELATION {type: $rel_type}]->(t)"
                )
                session.run(
                    query,
                    source=rel['source'],
                    target=rel['target'],
                    rel_type=rel['type']
                )

    def multi_hop_traversal(self, root_entity: str, hops: int = 2) -> List[Dict[str, Any]]:
        """
        Traverses the knowledge graph up to N hops from a root entity to gather connected context.
        """
        with self.driver.session() as session:
            # Cypher query for variable-length path traversal
            query = (
                f"MATCH path = (s:Entity {{name: $root}})-[:RELATION*1..{hops}]-(t:Entity) "
                "RETURN path"
            )
            result = session.run(query, root=root_entity)

            subgraphs = []
            for record in result:
                path = record["path"]
                nodes = [{"name": node["name"], "type": node.get("type")} for node in path.nodes]
                rels = [{"type": rel["type"]} for rel in path.relationships]
                subgraphs.append({"nodes": nodes, "relationships": rels})

            return subgraphs


if __name__ == "__main__":
    # Connect to local Docker Neo4j instance
    URI = "bolt://localhost:7687"
    USER = "neo4j"
    PASSWORD = "password123"

    client = Neo4jGraphClient(URI, USER, PASSWORD)

    # Mock enterprise entities extracted from architecture docs
    sample_entities = [
        {"name": "AuthService", "type": "Service"},
        {"name": "UserDatabase", "type": "Database"},
        {"name": "BillingService", "type": "Service"},
        {"name": "StripeAPI", "type": "ExternalAPI"},
        {"name": "DevOpsTeam", "type": "Team"}
    ]

    # Mock relationships representing dependencies
    sample_relations = [
        {"source": "AuthService", "target": "UserDatabase", "type": "DEPENDS_ON"},
        {"source": "BillingService", "target": "AuthService", "type": "CALLS"},
        {"source": "BillingService", "target": "StripeAPI", "type": "INTEGRATES_WITH"},
        {"source": "DevOpsTeam", "target": "AuthService", "type": "OWNS"}
    ]

    print("Ingesting Knowledge Graph into Neo4j...")
    client.add_document_kg(sample_entities, sample_relations)
    print("Ingestion complete!")

    # Test multi-hop traversal starting from 'BillingService'
    root = "BillingService"
    print(f"\nRunning 2-hop graph traversal for root entity: '{root}'...")
    subgraphs = client.multi_hop_traversal(root_entity=root, hops=2)

    for idx, sg in enumerate(subgraphs):
        path_str = " -> ".join([f"[{n['type']}] {n['name']}" for n in sg['nodes']])
        print(f"Path {idx + 1}: {path_str}")

    client.close()