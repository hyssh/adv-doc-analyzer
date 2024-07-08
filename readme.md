# Advanced Document Analyzer(MTC project ADA)

Goal of the sample app is to show case how to use GPT (LLMs) to analyze document

## Scenario

Document comparision using LLM for contract review works.

## Document comparison

One of common challenges are the document continuously changing. Especially when two different company making a deal or agreement, they exchange different version of the same or similar documents over and over.

By leveraging LLMs, people can reduce time to review document but increase productivity.

## Workflow

There are standard document for a contract. And when partner sends document of contract terms, the reviewer need to use the standard document and find differences between the standard and new documents.

## Azure Resouces

    Azure OpenAI
    Azure AI Search
    Azure Cognitive Service - Multiaccount
    Azure Web App

## Set Environment Variables

    AZURE_OPENAI_API_BASE=
    AZURE_OPENAI_API_KEY=
    AZURE_OPENAI_API_VERSION=
    DEPLOYMENT_NAME=
    EMBEDDING_MODEL_NAME=
    AZURE_STORAGE_CONNECTION_STRING=
    AZURE_FORM_RECOGNIZER_ENDPOINT=
    AZURE_FORM_RECOGNIZER_KEY=
    AZURE_SEARCH_ENDPOINT=
    AZURE_SEARCH_KEY=
    AZURE_SEARCH_API_VERSION=
    AZURESEARCH_FIELDS_CONTENT_VECTOR=contentVector

## Start

```bash
cd app
chainlit run chainlit_app.py
```
