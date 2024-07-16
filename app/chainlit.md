# Advanced Document Analyzer(MTC project ADA)

Goal of the sample app is to show case how to use GPT (LLMs) to analzye document

## Scenario

Document comparision using LLM for contract review works.

One of common challenges are the document continuously changing. Especially when two different company making a deal or agreement, they exchange different version of the same or similar documents over and over.

By leveraging LLMs, people can reduce time to review document but increase productivity.

## Supported features

- `Upload Document` - Upload a document for review, QA or comparison. This task will help you to create an Index in a Azure AI search.
- `Question and Answer` - Aka, Ask to your data. AI can help you to answer questions around a document. You must choose one document in the settings. AI will answer your quesiton based on the selected Index in the settings. This task will help you to find additional insights in a document.
- `Document Comparison` - AI can help you to compare two documents. You must choose two Indexes (documents) in the settings. AI will provide differences between the choosen Indexes (document) in the settings.
- `Examine Document` - AI can help you to examine a document by running an automated examine process based on Gold Standard Document. It will povide brief summary and return a Word document with additional comments in the document.

## Workflow

There are standard document for a contract. And when partner sends document of contract terms, the reviewer need to use the standard document and find differences between the standard and new documents.