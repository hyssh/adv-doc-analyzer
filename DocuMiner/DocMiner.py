from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import uuid4
from azure.ai.documentintelligence.models import AnalyzeResult 
from DocMinerPreprocess import DocMinerPreprocess


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