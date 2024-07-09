"""
This module is used to preprocess the data before document analysis

0. Initialize app
- Check access to services
- Create table for metadata and logging

1. Upload document
- Check if document is in the right format
- Upload to container in blob storage 
- Log the document metadata in table

2. Chunk the document
- Use Document Intelligence Layout
- Save the result from AI Document to blob storage
- Log the result in table
- Start to chunk the document
- Save the chunked document in blob storage
- Log the chunked document in table

3. Save the chunked document
- Get the chunked document from blob storage
- Get embeddings of the chunked document
- Save the embeddings in Azure AI search
- Log index name in table
"""

# create a table in azure table storage
import os
import logging  
import json
import base64
import openai
from openai import AzureOpenAI
import requests
import metadatamanager as metadata
from ProcessStage import ProcessStage
from document import DocumentMetadata
# from utils import Utils 
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
# from azure.ai.formrecognizer import DocumentAnalysisClient, DocumentAnalysisApiVersion
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, ContentFormat, AnalyzeResult 
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ComplexField,
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchAlgorithmMetric,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    VectorSearchProfile,
    # AzureOpenAIVectorizer,
    # AzureOpenAIParameters,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)

# check metadata table

class Preprocess:
    """
    Upload file to blob and log metadata    
    """
    # source_document: DocumentMetadata

    def __init__(self, is_user_doc: bool = True, 
                 user_document_index_name: str = None, 
                 container_name: str = "ada-container"):
        """
        Initialize the class
        """
        load_dotenv()
        # # Configure root logger to ignore DEBUG and INFO messages  
        logging.basicConfig(level=logging.WARNING)  
        logging.getLogger('azure').setLevel(logging.WARNING)  
        logging.getLogger('openai').setLevel(logging.WARNING)  

        # self.utils = Utils()
        assert self.check_azure_ai_search(), "Azure AI Search is not available"
        self.metadata_manager = metadata.MetadataManager()
        self.container_name = container_name
        self.source_document = DocumentMetadata(None)
        self.is_user_doc = is_user_doc
        self.user_document_index_name = user_document_index_name

    async def upload_document(self, source_file_full_path: str):
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
        blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))

        # check container exsitence
        # get file name from source_file_full_path
        file_name = os.path.basename(source_file_full_path)
        
        try:
            blob_service_client.create_container(self.container_name)
        except ResourceExistsError:
            pass

        blob_client = blob_service_client.get_blob_client(container=self.container_name, blob=file_name)

        with open(source_file_full_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
        
        # log the document metadata in table
        self.source_document = DocumentMetadata(file_name)
        self.source_document.source_file_fullpath = source_file_full_path
        self.source_document.blob_path = blob_client.url
        self.source_document.RowKey = ProcessStage.FILE_UPLOAD.name
        self.metadata_manager.insert_event(self.source_document.to_dict())

        

    async def chunk_and_save_document(self, analyzed_document: AnalyzeResult, container_name: str = 'ada-container-chunked'):
        """
        Chunk the document
        """
        # use Document Intelligence Layout
        # save the result from AI Document to blob storage
        # log the result in table
        # start to chunk the document
        # save the chunked document in blob storage
        # log the chunked document in table
        assert analyzed_document is not None, "Document analysis is required"
        assert self.source_document is not None, "Source document is required"
        
        blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))

        saved_file_lists = []

        try:
            # print(f"Creating container {container_name}")
            blob_service_client.create_container(container_name)
        except ResourceExistsError:
            pass

        # save each paragraph in a separate files
        for i, paragraph in enumerate(analyzed_document.paragraphs):
            paragraph_file_name = self.source_document.PartitionKey.replace(".docx", f"_paragraph_{i}.txt")
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=paragraph_file_name)
            blob_client.upload_blob(paragraph.content, overwrite=True)
            saved_file_lists.append(blob_client.url)

        # log the chunked document in table
        self.source_document.RowKey = ProcessStage.CHUNKING_END.name
        self.source_document.chunked_blob_path = container_name
        self.metadata_manager.insert_event(self.source_document.to_dict())

        # return saved_file_lists
    

    async def run_document_analysis(self):
        """
        Get the document analysis
        """
        # get the document analysis from blob storage
        # log the document analysis in table
        endpoint = os.getenv('AZURE_FORM_RECOGNIZER_ENDPOINT')
        key = os.getenv('AZURE_FORM_RECOGNIZER_KEY')

        blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
        blob_client = blob_service_client.get_blob_client(container=self.container_name, blob=self.source_document.PartitionKey)

        document_intelligence_client  = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

        document_bytes = blob_client.download_blob().readall()

        poller = document_intelligence_client.begin_analyze_document(model_id="prebuilt-layout",
                                                                    analyze_request=AnalyzeDocumentRequest(bytes_source=document_bytes),
                                                                    output_content_format=ContentFormat.TEXT)
        
        result: AnalyzeResult = poller.result()
        
        result_file_name = self.source_document.PartitionKey.replace(".docx","_doc_ai.json")
        # blob_upload_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
        upload_client = blob_service_client.get_blob_client(container=self.container_name, blob=result_file_name)
        upload_client.upload_blob(json.dumps(result.as_dict()), overwrite=True)

        # log the chunked document in table
        self.source_document.RowKey = ProcessStage.AI_DOCUMENT_ANALYSIS_END.name
        self.source_document.analyized_json = result_file_name
        self.metadata_manager.insert_event(self.source_document.to_dict())

        return result
    
    def text_to_base64(self, text):
        # Convert text to bytes using UTF-8 encoding
        bytes_data = text.encode('utf-8')

        # Perform Base64 encoding
        base64_encoded = base64.b64encode(bytes_data)

        # Convert the result back to a UTF-8 string representation
        base64_text = base64_encoded.decode('utf-8')

        return base64_text
    

    def check_azure_ai_search(self, index_name: str="ada_index_0",):
        credential = AzureKeyCredential(os.getenv('AZURE_SEARCH_KEY'))
        
        try:
            SearchClient(endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'), 
                                index_name=index_name, 
                                credential=credential)
            return True
        except Exception as e:
            print("Exception:",e)
            return False
            

    async def create_index_azure_search(self, index_name: str="ada_index_0"):
        credential = AzureKeyCredential(os.getenv('AZURE_SEARCH_KEY'))

        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchField(name="title", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="my-default-vector-profile"),
            SearchField(name="filepath", type=SearchFieldDataType.String, searchable=True, filterable=True),
            SearchField(name="url", type=SearchFieldDataType.String),
            SearchField(name="paragraph_num", type=SearchFieldDataType.Int32),
        ]

        # Example configuration (you can replace this with your data)
        # profile_name = "my-hnsw-config-1"
        # algorithm_config_name = "my-algorithms-config"

        # Configure the vector search configuration  
        vector_search = VectorSearch(  
            algorithms=[  
                HnswAlgorithmConfiguration(  
                    name="my-hnsw-config-1",  
                    parameters=HnswParameters(  
                        m=4,  
                        ef_construction=400,  
                        ef_search=500,  
                        metric=VectorSearchAlgorithmMetric.COSINE,  
                    ),  
                ),  
                ExhaustiveKnnAlgorithmConfiguration(  
                    name="my-knn-config-1",  
                    parameters=ExhaustiveKnnParameters(  
                        metric=VectorSearchAlgorithmMetric.COSINE,  
                    ),  
                ),  
            ],  
            profiles=[  
                VectorSearchProfile(  
                    name="my-default-vector-profile",  
                    algorithm_configuration_name="my-hnsw-config-1",  
                ),  
                VectorSearchProfile(  
                    name="myExhaustiveKnnProfile",  
                    algorithm_configuration_name="my-knn-config-1",  
                ),  
            ],  
        )

        semantic_config = SemanticConfiguration(  
            name="my-semantic-config",  
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content")]  
            ),  
        )

        semantic_search = SemanticSearch(configurations=[semantic_config])

        # Define the index
        index = SearchIndex(name=index_name, 
                            fields=fields,
                            vector_search=vector_search,
                            semantic_search=semantic_search)

        # Create the index
        search_index_client = SearchIndexClient(endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
                                          credential=credential)
        
        search_index_client.create_index(index)
 

    async def build_index_search(self, index_name: str = 'ada_index_0', container_name: str = 'ada-container-chunked'):
        """
        build an index for Azure AI Search
        """
        assert self.source_document is not None, "Source document is required"

        openai.api_type = "azure"
        openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")
        openai.api_version = "2023-03-15-preview"
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

        credential = AzureKeyCredential(os.getenv('AZURE_SEARCH_KEY'))
        client = SearchClient(endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'), 
                              index_name=index_name, 
                              credential=credential)
        
        openai_client = AzureOpenAI(api_key = os.getenv("AZURE_OPENAI_API_KEY"),
                                    api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
                                    azure_endpoint = os.getenv("AZURE_OPENAI_API_BASE"))

        blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
        # get a list of blobs using blob_service_client
        container_client = blob_service_client.get_container_client(container_name)
        blob_list = container_client.list_blobs()
        
        # read data in the blob
        for blob in blob_list:
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
            # print(blob_client.url)
            # get file name for the blob.url to use it as document_name
            filename = blob.name.split("/")[-1].split(".")[0]        
            paragraph_num = int(filename.split("_")[-1]) + 1
            blob_data = blob_client.download_blob().readall()
            blob_data = blob_data.decode("utf-8", "ignore")
            if len(blob_data) >= 30:
                title = blob_data[:30]
            else:
                title = blob_data
            try:
                upload_payload = {
                            "id": self.text_to_base64(filename),
                            "title": f"{title}",
                            "content": blob_data,
                            "contentVector": openai_client.embeddings.create(input=[blob_data], model="text-embedding-ada-002").data[0].embedding,
                            "filepath": filename,
                            "url": blob_client.url,
                            "paragraph_num": paragraph_num
                        }
                
                result = client.upload_documents(documents=[upload_payload])
        
                if not result[0].succeeded:
                    print(f"Document upload failed: {result[0].key} {result[0].status_code} {result[0].message}")
            
            except Exception as e:
                print("Exception:",e)
                break

        # log the chunked document in table
        self.source_document.RowKey = ProcessStage.NEW_INDEX_END.name
        self.metadata_manager.insert_event(self.source_document.to_dict())

        return self.source_document
