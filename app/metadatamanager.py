"""
This module is used to log the data for the app

"""

# create a table in azure table storage
import os
import uuid
from ProcessStage import ProcessStage
from document import DocumentMetadata
from azure.data.tables import TableClient
from azure.core.exceptions import ResourceExistsError
from azure.data.tables import TableServiceClient


# check metadata table
class MetadataManager:
    """
    This class is used to log the data for the app
    """

    # properties
    table_name: str
    document_metadata: DocumentMetadata

    def __init__(self, table_name: str = 'adametadata'):
        """
        Initialize the class
        """
        self.table_name = table_name
        self.check_table()

    # def __init__(self, table_name: str = 'ada-metadata', document_metadata: DocumentMetadata = None):
    #     """
    #     Initialize the class
    #     """
    #     self.table_name = table_name
    #     self.document_metadata = document_metadata

    def set_file_name(self, file_name: str):
        """
        Set the file name for the metadata
        """
        self.document_metadata = DocumentMetadata(file_name)

    def set_raw_key(self, process_stage: ProcessStage):
        """
        Set the raw key for the metadata
        """
        self.document_metadata.RowKey = process_stage

    def set_blob_path(self, blob_path: str):
        """
        Set the blob path for the metadata
        """
        self.document_metadata.blob_path = blob_path

    def set_chunked_blob_path(self, chunked_blob_path: str):
        """
        Set the chunked blob path for the metadata
        """
        self.document_metadata.chunked_blob_path = chunked_blob_path

    def get_source_file_fullpath(self):
        """
        Get the file name for the metadata
        """
        return self.file_name

    def get_blob_path(self):
        """
        Get the blob path for the metadata
        """
        return self.blob_path

    def get_chunked_blob_path(self):
        """
        Get the chunked blob path for the metadata
        """
        return self.chunked_blob_path
        
    def check_table(self):
        """
        Check if the table exists in the storage account
        """
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        table_name = self.table_name

        # create a table client

        
        # check if the table exists        
        try:
            table_client = TableServiceClient.from_connection_string(connection_string)
            # table_client.create_table()
            print(f"Trying to create a table {self.table_name}")
            table_client.create_table(table_name)
        except ResourceExistsError:
            print(f"{self.table_name} already exists")
    
    def insert_event(self, event_data: dict):
        """
        Insert an event into the table
        """
        # get the connection string
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')        
        # create a table client
        table_client = TableClient.from_connection_string(connection_string, self.table_name)
        
        # insert the event
        table_client.upsert_entity(entity=event_data)
            
    def get_events(self):
        """
        Get all events from the table
        """
        # get the connection string
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        
        # create a table client
        table_client = TableClient.from_connection_string(connection_string, self.table_name)
        
        # get all entities
        entities = table_client.list_entities()
       
        return entities
    

