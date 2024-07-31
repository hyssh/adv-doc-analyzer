# Advanced Document Analyzer

Goal of the sample app is to show case how to use GPT (LLMs) to analzye document

## How to use Document Comparison

1. [Download a sample word document](https://raw.githubusercontent.com/hyssh/adv-doc-analyzer/master/sample_docs/test.docx) for test
1. Make any changes to the document
1. Save the updated docuemnt
1. **Click** `Upload Document` and give a name, e.g., test123
1. Wait until it finish the indexing
1. Start new chat and **Choose** task you want to test
1. E.g., `Document Comparison` need to select **Two**  indexes and you can ask questions to compare the selected two documents

## How to use Examine Document

1. [Download a sample word document](https://raw.githubusercontent.com/hyssh/adv-doc-analyzer/master/sample_docs/test.docx) for test
1. Make any changes to the document
1. Save the updated docuemnt
1. **Click** `Examine Document`
1. Upload your word document
1. Wait until it finish the examination
1. Review the response and download commented word document

## Supported features

- `Upload Document` - Upload a document for review, QA or comparison. This task will help you to create an Index in a Azure AI search.
- `Question and Answer` - Aka, Ask to your data. AI can help you to answer questions around a document. You must choose one document in the settings. AI will answer your quesiton based on the selected Index in the settings. This task will help you to find additional insights in a document.
- `Compare Documents` - AI can help you to compare two documents. You must choose two Indexes (documents) in the settings. AI will provide differences between the choosen Indexes (document) in the settings.
- `Examine Document` - AI can help you to examine a document by running an automated examine process based on Gold Standard Document. It will povide brief summary and return a Word document with additional comments in the document.

## Prompts

### Comparison

```markdown
## Instruction
If the question is not about comparing the document, please provide the below response:
Make sure both Gold Standard and Review documents have similar context, if not inform the user about the differences.
If the documents are completely off topic each other, out of contract or not related, inform the user about the situation.
Follow the below steps to answer the user question:        
### 1. Check Gold Stanard document 
If there is no Gold Standard document available in the retrieved data, inform the user ask the question a different way.
If there are retrived data in the Gold Standard document provide summary of the document.
### 2. Check Review document
If there are retrived data in the Gold Standard document but not Review document then it maybe the terms and/or conditions were removed or not available in the Review document.
If there are retrived data in the Review document provide summary of the document.
### 3. Compare the Gold standard document and the Review document
Compare the Gold Standard document and the Review document to find the differences.
Find differences between the documents and summaryize the differences.
Make sure include the number of sections and section names in the summary.
### 4. Answer the user question based on the comparison
If there are no differences between the documents, inform the user that there are no differences. 
Do not provide or information that are the same.
It is important to provide the differences between the documents.
Explain differences between the Gold Standard document and the Review document.
Add commnets risks that may arise from the differences.   
```