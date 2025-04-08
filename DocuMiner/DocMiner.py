from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import uuid4
from azure.ai.documentintelligence.models import AnalyzeResult 
import networkx as nx  
import matplotlib.pyplot as plt  

"""
# Example usage
doc = DocMiner(
    document_filename="example.pdf",
    project_title="Sample Project",
    author="John Doe",
    table_of_contents=["Introduction", "Chapter 1", "Chapter 2"],
    keywords=["sample", "project", "example"]
)

print(doc.document_id)
"""
class Task(BaseModel):
    task_code: str
    task_description: str

class DocMiner(BaseModel):    
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    document_filename: str
    file_location: Optional[str] = None
    project_title: str
    author: str = None
    table_of_contents: List[str]
    keywords: List[str]
    tasks: Optional[List[Task]] = None
    reference_documents: Optional[List[str]] = None

    @staticmethod  
    def create_graph(documents: List['DocMiner']) -> nx.DiGraph:  
        G = nx.DiGraph()  
  
        # Add nodes and edges for documents  
        for doc in documents:  
            G.add_node(doc.document_filename, label=doc.project_title, document_id=doc.document_id, tasks=doc.tasks)  
            if doc.reference_documents:  
                for ref in doc.reference_documents:  
                    G.add_edge(doc.document_filename, ref)  
  
        # Optionally add task nodes and edges if you want to visualize them separately  
        for doc in documents:  
            if doc.tasks:  
                for task in doc.tasks:  
                    task_node = f"{doc.document_filename}_{task.task_code}"  
                    G.add_node(task_node, label=task.task_code, document_id=doc.document_id)  
                    G.add_edge(doc.document_filename, task_node)  
  
        return G  
  
    @staticmethod  
    def get_document_ids_by_task_codes(G, task_code) -> List[str]:  
        document_ids = set()  
        for node, data in G.nodes(data=True):  
            if 'tasks' not in data:  
                continue  
            tasks = data['tasks']  
            if any(task.task_code == task_code for task in tasks):  
                document_ids.add(data['document_id'])  
        return list(document_ids)  
  
    @staticmethod
    def get_node_by_id(G, document_id):
        for node, data in G.nodes(data=True):
            if data['document_id'] == document_id:
                return node
        return None

    @staticmethod  
    def find_top_document_id(G):  
        in_degrees = dict(G.in_degree())  
        top_document = max(in_degrees, key=in_degrees.get)  
        document_id = G.nodes[top_document]['document_id']  
        return document_id  
  
    @staticmethod      
    def draw_graph(G) -> None:  
        pos = nx.spring_layout(G)  
        plt.figure(figsize=(16, 12))  
  
        # Draw the nodes and edges  
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue', node_shape='o')  
        nx.draw_networkx_edges(G, pos, arrows=True)  
  
        # Draw labels for document nodes  
        document_labels = {node: data['label'] for node, data in G.nodes(data=True) if 'label' in data and 'tasks' in data}  
        nx.draw_networkx_labels(G, pos, labels=document_labels, font_size=10, font_weight='bold')  
  
        # Draw labels for task nodes  
        task_labels = {node: data['label'] for node, data in G.nodes(data=True) if 'label' in data and 'tasks' not in data}  
        nx.draw_networkx_labels(G, pos, labels=task_labels, font_size=8, font_color='red')  
  
        plt.title('Document Reference Graph with Tasks')  
        plt.show()  
