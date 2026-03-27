import networkx as nx
import json
import pickle
from community import community_louvain
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from config.settings import RAGConfig
from .generator import ResponseGenerator
from models.embeddings import EmbeddingModel


@dataclass
class EntityNode:
    """
    Represents an entity in the knowledge graph.

    For each entity node, the following are attached:
    - id (str): identifier for the entity
    - name (str): name of the entity node
    - description (str): description of the entity node
    - source_chunks (List[int]): chunks from the source document for this entity, identified by chunk IDs
    - degree (int): number of relationship edges connected to this entity node
    - community_id (int): identifier for the community this node is in
    """
    id: str
    name: str
    description: str
    source_chunks: List[int]
    degree: int = 0
    community_id: int = None


@dataclass
class RelationshipEdge:
    """
    Represents a relationship between entities as a graph edge.

    For each graph relationship, the following are attached:
    - source (str): identifier (id) of source entity node
    - target (str): identifier (id) of target entity node
    - relationship (str): relationship between the connected nodes
    - description (str): description of the relationship
    - weight (int): weight of the relationship edge
    - source_chunks (List[int]): chunks from the source document for this relationship, identified by chunk IDs
    """
    source: str
    target: str
    relationship: str
    description: str
    weight: int = 1
    source_chunks: Optional[List[int]] = None


class GraphStorageManager:
    """
    Manages graph-based storage and retrieval using knowledge graphs.
    """
    
    def __init__(self, config: RAGConfig):
        """
        Initializes an instance of the graph storage manager class for graph structures and storage paths.

        args:
        - config (RAGConfig): an instance of the data class for configuration settings
        """
        self.config = config
        self.generator = ResponseGenerator(config=config)
        self.embedding = EmbeddingModel(config=config)
        self.graph = nx.Graph()
        self.entities = {}                                  # maps entity ids to EntityNode objects
        self.relationships = []                             # a list of RelationshipEdge objects
        self.communities = {}                               # maps community ids to lists of entity ids
        self.community_summaries = {}                       # maps community ids to community summaries
        self.community_summary_embeddings = {}              # maps community ids to vector embeddings of community summaries
        self.chunk_to_entities = {}                         # maps chunk ids to lists of entity ids
        self.persist_path = config.graph_persist_directory

        os.makedirs(self.persist_path, exist_ok=True)       # creates directory for graph storage

    def extract_entities_and_relationships(self, chunks: List[str], metadatas: List[Dict]) -> Tuple[Dict, List]:
        """
        Extract entities and relationships from text chunks using LLM.
        
        args:
        - chunks (List[str]): chunks of text from source documents
        - metadatas (List[Dict]): metadatas corresponding to chunks

        returns:
        - a tuple consisting of a dictionary of entities and a list of relationships
        """
        entities, relationships = {}, []

        for i, (chunk, metadata) in enumerate(zip(chunks, metadatas)):
            chunk_id = metadata.get("chunk_index", i)

            prompt = self._build_entity_extraction_prompt(text=chunk)
            response = self.generator.generate_openai_response(prompt=prompt, query=chunk)

            try:
                response_output = response.get("answer", "Error generating response.")

                if "Error generating response" in response_output:
                    print("An error occurred in extracting entities and relationships.")
                    continue
                
                parsed_response = self._parse_extraction_response(response_text=response_output)

                for entity in parsed_response.get("entities", []):
                    entity_id = f"{entity["name"]}_{hash(entity["name"]) % 10000}"      # hash to get key for efficient lookup

                    if entity_id not in entities:
                        entities[entity_id] = EntityNode(
                            id=entity_id,
                            name=entity["name"],
                            description=entity.get("description", ""),
                            source_chunks=[chunk_id]
                        )

                    else:
                        entities[entity_id].source_chunks.append(chunk_id)              # node exists, simply add chunk id to the node

                for relation in parsed_response.get("relationships", []):
                    source_id = f"{relation["source"]}_{hash(relation["source"]) % 10000}"
                    target_id = f"{relation["target"]}_{hash(relation["target"]) % 10000}"

                    relationships.append(RelationshipEdge(
                        source=source_id,
                        target=target_id,
                        relationship=relation.get("relationship", ""),
                        description=relation.get("description", ""),
                        source_chunks=[chunk_id]
                    ))

                    # track which entities are related to and mentioned in this chunk
                    self.chunk_to_entities.setdefault(chunk_id, []).extend([source_id, target_id])

            except Exception as e:
                print(f"An error occurred in extracting entities and relationships: {e}")
                continue
        
        return entities, relationships

    def _build_entity_extraction_prompt(self, text: str) -> str:
        """
        Builds prompt for entity and relationship extraction.

        args:
        - text (str): the text to extract entities and relationships from

        returns:
        - a prompt for extracting entities and relationships from the text
        """
        prompt = f"""You are an intelligent document processor, and you will assist in storing document information efficiently.
The information below is to be parsed into entities and relationships between them.

# Provided Information
{text}

# Instructions
Read the information and extract entities and relationships between them from the text.

## Entities
For entities you find in the text, provide its **name** and **a short description** of what it is.

## Relationships
For relationships between two entities you find in the text, provide the **source entity**, **target entity**,
the **relationship** between them (relationship type), and their **a short description**.

# Output Format
Give your output in a JSON string similar to the following.
Do **NOT** include any markdown backticks in your output.
Remember that your output will be directly parsed into a Python object.

## Example Output
{{
  "entities": [
    {{"name": "Entity1", "description": "Description of entity1"}},
    {{"name": "Entity2", "description": "Description of entity2"}}
  ],
  "relationships": [
    {{"source": "Entity1", "target": "Entity2", "relationship": "influences", "description": "How entity1 influences entity2"}}
  ]
}}

Response:
"""
        return prompt

    def _parse_extraction_response(self, response_text: str) -> Dict:
        """
        Parses LLM response for entity extraction.

        args:
        - response_text (str): response from LLM to be parsed

        returns:
        - a Python dict containing entities and relations extracted from the chunked text
        """
        try:
            if "```json" in response_text:
                # remove potential markdown backticks
                response_text = response_text.split("```json")[1].split("```")[0]

            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")

            if start_idx != -1 and end_idx != -1:
                # non-empty substring found between quotes
                json_str = response_text[start_idx:end_idx+1]
                parsed = json.loads(json_str)

                return parsed

        except:
            pass

        # response string cannot be parsed, return empty dict
        return {"entities": [], "relationships": []}


    def build_knowledge_graph(self, entities: Dict[str, EntityNode], relationships: List[RelationshipEdge]):
        """
        Builds knowledge graph from entities and relationships.
        
        args:
        - entities (Dict[str, EntityNode]): a dictionary mapping node ids to their corresponding objects
        - relationships (List[RelationshipEdge]): a list of RelationshipEdge objects depicting relations between different entities
        """
        self.graph.clear()
        self.entities = entities
        self.relationships = relationships

        for entity_id, entity in entities.items():
            self.graph.add_node(
                entity_id,
                name=entity.name,
                description=entity.description,
                source_chunks=entity.source_chunks
            )

        for relation in relationships:
            if relation.source in self.graph.nodes() and relation.target in self.graph.nodes():
                # only when both the source node and target node exist
                if self.graph.has_edge(relation.source, relation.target):
                    # relation alredy exists, increment weight and append description
                    self.graph[relation.source][relation.target]["weight"] += relation.weight
                    self.graph[relation.source][relation.target]["description"].append(relation.description)
                else:
                    self.graph.add_edge(
                        relation.source,
                        relation.target,
                        weight=relation.weight,
                        description=[relation.description],
                        relationship=relation.relationship
                    )

        for node in self.graph.nodes():
            # update degree of each node after processing the graph
            self.entities[node].degree = self.graph.degree(node)

    def detect_communities(self):
        """
        Detects communities in the knowledge graph using Louvain algorithm.
        """
        partition = community_louvain.best_partition(self.graph)
        
        self.communities = {}
        for node, community_id in partition.items():
            self.communities.setdefault(community_id, []).append(node)
            self.entities[node].community_id= community_id

        self.communities = {    # filter out communities that are too small
            community_id: nodes for community_id, nodes in self.communities.items() if len(nodes) >= self.config.min_community_size
        }

    def generate_community_summaries(self):
        """
        Generates summaries for each community using an LLM.
        """
        for community_id, nodes in self.communities.items():
            subgraph = self.graph.subgraph(nodes)
            community_info = self._collect_community_info(subgraph=subgraph, nodes=nodes)

            prompt = self._build_community_summary_prompt(community_info=community_info)

            response = self.generator.generate_openai_response(prompt=prompt, query="")

            try:
                summary = response.get("answer", "Error generating response.")

            except Exception as e:
                summary = f"Error generating response: {e}"
                print(summary)
            
            if "Error generating response" in summary:
                summary = f"Community with {community_info["size"]} entities."

            self.community_summaries[community_id] = summary

            summary_embedding = self.embedding.encode_single(summary)
            self.community_summary_embeddings[community_id] = summary_embedding

    def _build_community_summary_prompt(self, community_info: Dict) -> str:
        """
        Builds prompt for summarizing entities and relationships within a community.

        args:
        - community_info (Dict): a dictionary containing information collected from the community

        returns:
        - a prompt for summarizing the community's entities and relationships
        """
        prompt = f"""You are a helpful assistant that will assist with data handling tasks.
Previously some documents were processed, and entities and their relationships were extracted to form a knowledge graph.
Some communities were found in the knowledge graph, and your help is needed in organizing these communities.

Summarize the following community of related entities in about {self.config.community_summary_length} words.

Main entities:
{", ".join(f"{node["name"]}" for node in community_info["nodes"])}

Key relationships:
{"; ".join(f"{relation["source"]} {relation["relationship"]} {relation["target"]}" for relation in community_info["relationships"])}

Provide a concise summary that captures the main themes and relationships within this community.
"""
        return prompt

    def _collect_community_info(self, subgraph: nx.Graph, nodes: List[str]) -> Dict:
        """
        Collects information about a community.
        
        args:
        - subgraph (nx.Graph): subgraph containing the nodes in the community
        - nodes (List[str]): a list of node ids in the community

        returns:
        - a dictionary containing information about the community
        """
        node_degrees = [(node, subgraph.degree(node)) for node in nodes]
        node_degrees.sort(key=lambda x: x[1], reverse=True)         # sort nodes by node degrees
        top_nodes = node_degrees[:10]                               # retrieve top 10 entities

        relationships = []

        for u, v, data in subgraph.edges(data=True):
            description_list = data.get("description", [])
            description = description_list[0] if description_list else "No description"

            relationships.append({
                "source": self.entities[u].name,
                "target": self.entities[v].name,
                "relationship": data.get("relationship", "related"),
                "description": description
            })

        information = {
            "nodes": [{"name": self.entities[node].name, "degree": deg} for node, deg in top_nodes],
            "relationships": relationships[:10],                    # top 10 relationships
            "size": len(nodes)
        }

        return information

    def add_documents(self, chunks: List[str], metadatas: List[dict]):
        """
        Processes documents and builds knowledge graph.
        
        args:
        - chunks (List[str]): chunks of text from source documents
        - metadatas (List[Dict]): metadatas corresponding to chunks
        """
        entities, relationships = self.extract_entities_and_relationships(chunks=chunks, metadatas=metadatas)
        
        self.build_knowledge_graph(entities=entities, relationships=relationships)

        self.detect_communities()

        self.generate_community_summaries()

        self.save()

    def save(self):
        """
        Saves graph data to disk.
        """
        data = {
            "graph": self.graph,
            "entities": self.entities,
            "relationships": self.relationships,
            "communities": self.communities,
            "community_summaries": self.community_summaries,
            "community_summary_embeddings": self.community_summary_embeddings,
            "chunk_to_entities": self.chunk_to_entities
        }
    
        # save in binary format
        with open(os.path.join(self.persist_path, "graph_data.pkl"), "wb") as f:
            pickle.dump(data, f)

        data_json = {
            "entities": {eid: {"name": ent.name, "description": ent.description} for eid, ent in self.entities.items()},
            "communities": self.communities,
            "community_summaries": self.communities
        }

        # save in JSON format for readability
        with open(os.path.join(self.persist_path, "graph_data.json"), "w") as f:
            json.dump(data_json, f)

    def load(self) -> bool:
        """
        Loads graph data from disk.

        returns:
        - a boolean value indicating whether the load was successful
        """
        try:
            with open(os.path.join(self.persist_path, "graph_data.pkl"), "rb") as f:
                data = pickle.load(f)

            self.graph = data["graph"]
            self.entities = data["entities"]
            self.relationships = data["relationships"]
            self.communities = data["communities"]
            self.community_summaries = data["community_summaries"]
            self.community_summary_embeddings = data["community_summary_embeddings"]
            self.chunk_to_entities = data["chunk_to_entities"]

            return True
        
        except Exception as e:
            print(f"An error occurred while loading graph data from disk: {e}")
            return False            

    def search_by_entity(self, query: str, n_results: int = None) -> Dict:
        """
        Searches for entities matching query.

        args:
        - query (str): entity to search for
        - n_results (int): number of results to return

        returns:
        - a dictionary containing the matched entities and community summaries
        """
        if n_results is None:
            n_results = self.config.top_k

        matching_entities = []
        for entity_id, entity in self.entities.items():
            if query.lower() in entity.name.lower() or query.lower() in entity.description.lower():
                # keyword matching to search for entities
                matching_entities.append({
                    "id": entity_id,
                    "name": entity.name,
                    "description": entity.description,
                    "community": entity.community_id,
                    "degree": entity.degree
                })

        relevant_communities = set()
        for entity in matching_entities:
            if entity["community"] is not None:
                relevant_communities.add(entity["community"])

        community_results = []
        for community_id in list(relevant_communities)[:n_results]:
            # retrieve n_results community summaries
            if community_id in self.community_summaries:
                community_results.append({
                    "community_id": community_id,
                    "summary": self.community_summaries[community_id],
                    "entities": [self.entities[eid].name for eid in self.communities.get(community_id, [])[:5]]
                })

        query_results = {
            "matching_entities": matching_entities[:n_results],
            "relevant_communities": community_results,
            "total_entities_found": len(matching_entities)
        }

        return query_results

    def search_global_question(self, question: str) -> Dict:
        """
        Handles global/sensemaking questions.
        
        args:
        - question (str): question about the entire dataset

        returns:
        - a dictionary containing relevant communities and their summaries
        """
        question_embedding = self.embedding.encode_single(question)

        all_communities = []
        for community_id, community_summary in self.community_summaries.items():
            summary_embedding = self.community_summary_embeddings[community_id]
            similarity = self.embedding.embedding_similarity(question_embedding, summary_embedding)

            all_communities.append({
                "community_id": community_id,
                "community_summary": community_summary,
                "size": len(self.communities.get(community_id, [])),
                "entities": [self.entities[eid].name for eid in self.communities.get(community_id, [])[:3]],
                "similarity": similarity
            })

        all_communities.sort(key=lambda x: x["similarity"], reverse=True)         # sort communities in decreasing order of similarity

        search_results = {
            "question_type": "global",
            "communities": all_communities[:self.config.top_k],
            "total_communities": len(self.community_summaries)
        }

        return search_results