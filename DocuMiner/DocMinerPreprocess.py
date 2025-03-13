import os
import logging  
import json
from uuid import uuid4
from typing import Tuple
from urllib.parse import unquote
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, ContentFormat, AnalyzeResult 
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

# check metadata table
class DocMinerPreprocess:
    """
    Upload file to blob and log metadata    
    """
    def __init__(self,
                 container_name: str = "docminer-container"):
        """
        Initialize the class
        """
        load_dotenv()
        # # Configure root logger to ignore DEBUG and INFO messages  
        # logging.basicConfig(level=logging.WARNING)  
        # logging.getLogger('azure').setLevel(logging.WARNING)
        self.blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
        self.anlysis_result: AnalyzeResult = None
        self.chunked_blob_path = None
        self.container_name = container_name
        self.file_name = None

    def preprocess_document(self, source_file_full_path: str) -> AnalyzeResult:
        blob_path = self.upload_document(source_file_full_path) 
        self.anlysis_result, self.chunked_blob_path = self.run_save_document_analysis(blob_path)
        self.chunk_and_save_document(self.anlysis_result)
        return self.anlysis_result, unquote(blob_path)
        
    def upload_document(self, source_file_full_path: str) -> str:
        """
        Upload document to blob storage
        """
        # check if the document is in the right format
        source_file_full_path = os.path.abspath(source_file_full_path)

        if not os.path.exists(source_file_full_path):
            raise Exception('Document does not exist')

        if not source_file_full_path.endswith('.docx'):
            raise Exception('Document is not in docx format')
        
        # upload the document to blob storage
        # blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))

        # check container exsitence
        # get file name from source_file_full_path
        self.file_name = os.path.basename(source_file_full_path)
        
        try:
            self.blob_service_client.create_container(self.container_name)
        except ResourceExistsError:
            pass

        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=self.file_name)

        with open(source_file_full_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
        
        # retrun blob storage location
        return unquote(blob_client.url)
        

    def run_save_document_analysis(self, blob_url: str) -> Tuple[AnalyzeResult, str]:
        """
        Get the document analysis
        """
        # get the document analysis from blob storage
        # log the document analysis in table
        endpoint = os.getenv('AZURE_FORM_RECOGNIZER_ENDPOINT')
        key = os.getenv('AZURE_FORM_RECOGNIZER_KEY')

        # blob_url = 'https://openaiembedding.blob.core.windows.net/docminer-container/Plan%20for%20Model%20Training%20Using%20Azure%20Machine%20Learning%20Service_V0.docx'
        blob_url = unquote(blob_url)
        container_name = blob_url.split('/')[3]
        blob_name = blob_url.split('/')[-1]
        blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        document_intelligence_client  = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        document_bytes = blob_client.download_blob().readall()

        poller = document_intelligence_client.begin_analyze_document(model_id="prebuilt-layout",
                                                                    analyze_request=AnalyzeDocumentRequest(bytes_source=document_bytes),
                                                                    output_content_format=ContentFormat.TEXT)
        
        result: AnalyzeResult = poller.result()
        
        result_file_name = blob_name.replace(".docx","_doc_ai.json")
        upload_client = self.blob_service_client.get_blob_client(container=container_name, blob=result_file_name)
        upload_client.upload_blob(json.dumps(result.as_dict()), overwrite=True)

        return result, unquote(upload_client.url)
        

    def chunk_and_save_document(self, analyzed_document: AnalyzeResult, container_name: str = 'docminer-container-chunked'):
        """
        Chunk the document
        """
        # use Document Intelligence Layout
        # save the result from AI Document to blob storage
        # log the result in table
        # start to chunk the document
        # save the chunked document in blob storage
        assert analyzed_document is not None, "Document analysis is required"
        
        # blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))

        saved_file_lists = []

        try:
            # Creating container for chucnked document
            self.blob_service_client.create_container(container_name)
        except ResourceExistsError:
            # print(f"{container_name} is found")
            pass
        
        if self.file_name is None:
            self.file_name = str(uuid4())+".docx"

        # save each paragraph in a separate files
        for i, paragraph in enumerate(analyzed_document.paragraphs):
            paragraph_file_name = self.file_name.replace(".docx", f"_paragraph_{i}.txt")
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=paragraph_file_name)
            blob_client.upload_blob(paragraph.content, overwrite=True)
            saved_file_lists.append(blob_client.url)

        for page in analyzed_document.pages:
            # extract table from page
            for table_idx, table in enumerate(analyzed_document.tables):
                # check page number
                markdown_table = ""
                
                data = table.cells
                # Convert data to 2D list
                table = [["" for _ in range(max(cell.column_index for cell in data) + 1)] for _ in range(max(cell.row_index for cell in data) + 1)]  
                for cell in data:  
                    table[cell.row_index][cell.column_index] = cell.content  
                
                # Convert 2D list to markdown  
                markdown_table = ["| " + " | ".join(row) + " |" for row in table]  
                header_seperator = ["|---" * len(table[0]) + "|"]  
                markdown_table = markdown_table[:1] + header_seperator + markdown_table[1:]  
                
                markdown_table = "\n".join(markdown_table)

                # save markdown table to blob storage
                table_file_name = self.file_name.replace(".docx", f"_table_{table_idx}.md")
                blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=table_file_name)
                blob_client.upload_blob(markdown_table, overwrite=True)
        