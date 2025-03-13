from pydantic import BaseModel
from typing import List, Dict, Optional

class Task(BaseModel):
    TaskCode: str
    TaskDescription: str

class Node(BaseModel):
    id: str
    tasks: List[Task]

class Edge(BaseModel):
    source: str
    target: str
    relationship: str

class Graph(BaseModel):
    nodes: Dict[str, Node]
    edges: List[Edge]
    task_index: Dict[str, List[str]] = {}

    def __init__(self, **data):
        super().__init__(**data)
        self.build_index()

    def build_index(self):
        for node_id, node in self.nodes.items():
            for task in node.tasks:
                if task.TaskCode not in self.task_index:
                    self.task_index[task.TaskCode] = []
                self.task_index[task.TaskCode].append(node_id)

    def update_task(self, old_task_code: str, new_task_code: Optional[str] = None, new_description: Optional[str] = None):
        if old_task_code in self.task_index:
            for node_id in self.task_index[old_task_code]:
                node = self.nodes[node_id]
                for task in node.tasks:
                    if task.TaskCode == old_task_code:
                        if new_task_code:
                            task.TaskCode = new_task_code
                        if new_description:
                            task.TaskDescription = new_description
            if new_task_code:
                self.task_index[new_task_code] = self.task_index.pop(old_task_code)
