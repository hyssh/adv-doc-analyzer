import datetime
from typing import List
from ProcessStage import ProcessStage


class DocumentMetadata:
    # class properties
    PartitionKey: str
    RowKey: str
    source_file_fullpath: str
    blob_path: str
    chunked_blob_path: str
    created_datetime: datetime

    def __init__(self, document_file_name: str):
        """
        Initialize the class
        """
        self.PartitionKey = document_file_name
        self.RowKey = ProcessStage.INIT
        self.source_file_fullpath = document_file_name
        self.blob_path = None
        self.chunked_blob_path = None
        self.created_datetime = datetime.datetime.now()
    
    def to_dict(self):
        """
        Convert the class to a dictionary
        """
        return {
            'PartitionKey': self.PartitionKey,
            'RowKey': self.RowKey,
            'source_file_fullpath': self.source_file_fullpath,
            'blob_path': self.blob_path,
            'chunked_blob_path': self.chunked_blob_path,
            'created_datetime': datetime.datetime.now()
        }