
# Local Knowledge LLM & Agentic RAG

This repository demonstrates an Agentic Retrieval-Augmented Generation (RAG) system using a local Language Model (LLM). The system leverages LangChain, a powerful tool for building LLM applications, along with various other libraries to load, process, and retrieve information from PDFs.

## Overview

The Local Knowledge Agentic RAG system is designed to provide relevant information from a local knowledge base (PDF documents) using a local LLM. This system can be used for question-answering tasks where the answers are derived from the provided documents. It features a LangGraph workflow to manage the retrieval, grading, and generation processes, ensuring that the answers are relevant and grounded in the provided documents.

## Prerequisites

To run the Local Knowledge Agentic RAG system, you need to have a local LLM set up. This can be done using Ollama or any other model that fits your system requirements. You can start your local LLM using the following command:

```sh
ollama run {model_name}
```

## Setting Up

1. **Install the required dependencies**: Run the following command to install all necessary libraries.

    ```sh
    !pip install langchain-nomic langchain_community tiktoken langchainhub chromadb langchain langgraph tavily-python gpt4all firecrawl-py pymupdf langchain-ollama
    ```

2. **Setting up LangSmith, this helps you to view the work details**: Update the `local_agentic_rag.ipynb` notebook with your `LANGSMITH_API_KEY`.

    ```python
    import os

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGSMITH_API_KEY'] = 'your langsmith api key'
    ```

3. **Prepare your data**: Place the PDF documents you want to use in the `data` folder.

## Agentic RAG and LangGraph

### Agentic RAG

The Agentic RAG system involves several key steps:

- **Document Loading**: Load all PDF documents from the `data` folder using the `FileSystemBlobLoader` and `PyMuPDFParser`.
  
  ```python
  from langchain_community.document_loaders import FileSystemBlobLoader
  from langchain_community.document_loaders.generic import GenericLoader
  from langchain_community.document_loaders.parsers import PyMuPDFParser

  loader = GenericLoader(
      blob_loader=FileSystemBlobLoader(
          path='./data',
          glob='*.pdf',
      ),
      blob_parser=PyMuPDFParser(),
  )

  docs = loader.load()
  print(len(docs))
  ```

- **Text Splitting**: Split the loaded documents into manageable chunks using the `RecursiveCharacterTextSplitter`.

  ```python
  from langchain.text_splitter import RecursiveCharacterTextSplitter

  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
      chunk_size=500, chunk_overlap=0,
  )

  doc_splits = text_splitter.split_documents(docs)
  ```

- **Filtering Documents**: Filter out documents with less than 7 words and clean metadata.

  ```python
  from langchain.docstore.document import Document

  filtered_doc = []

  for doc in doc_splits:
      if isinstance(doc, Document) and hasattr(doc, 'metadata'):
          if len(doc.page_content) < 7:
              continue  # Skip this document if it has less than 7 words
          clean_metadata = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
          filtered_doc.append(Document(page_content=doc.page_content, metadata=clean_metadata))
  ```

- **Vector Store**: Store the filtered documents in a vector store using `Chroma` and `GPT4AllEmbeddings`.

  ```python
  from langchain_community.vectorstores import Chroma
  from langchain_community.embeddings import GPT4AllEmbeddings

  vectorstore = Chroma.from_documents(
      documents=filtered_doc,
      collection_name='lk_rag',
      embedding=GPT4AllEmbeddings(),
  )

  retriever = vectorstore.as_retriever(
      search_type="similarity_score_threshold",
      search_kwargs={"k": 5, "score_threshold": 0.5},
  )
  ```

### LangGraph Workflow

The LangGraph workflow orchestrates the entire process from document retrieval to answer generation, ensuring a structured and efficient flow. The workflow is built using several nodes, each responsible for specific tasks:

- **Retrieve Node**: The entry point of the workflow, where the system retrieves relevant documents from the vector store based on the user's query.
- **Grade Documents Node**: This node grades the retrieved documents for relevance. It uses a grading mechanism to filter out irrelevant documents, marking them for potential web search if necessary.
- **Generate Node**: If the documents are deemed relevant, this node generates an answer based on the retrieved documents.
- **Web Search Node**: If the documents are not sufficient, this node performs a web search to gather additional information.

The workflow includes decision points to determine the next steps:
- **Decide to Generate**: This decision point assesses whether the retrieved documents are sufficient to generate an answer or if a web search is required.
- **Is Hallucination and Useful**: This decision point checks if the generated answer is grounded in the provided documents and is useful for the query.

The workflow proceeds as follows:
1. The **Retrieve Node** fetches documents related to the query.
2. The **Grade Documents Node** evaluates the relevance of these documents.
3. Depending on the grading, the **Decide to Generate** point directs the flow either to the **Generate Node** or the **Web Search Node**.
4. If web search is needed, the **Web Search Node** gathers additional information and the flow returns to the **Generate Node**.
5. The **Generate Node** produces the final answer, which is then evaluated for grounding and usefulness by the **Is Hallucination and Useful** decision point.
6. Based on this evaluation, the workflow either concludes or loops back to the web search for more information.

This structured approach ensures that the answers generated are both relevant and reliable, leveraging both local knowledge and external resources when necessary.

```python
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph

class GraphState(TypedDict):
    question: str
    answer: str
    web_search: bool
    documents: List[str]

workflow = StateGraph(GraphState)

workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    is_hallucination_and_useful,
    {
        "useful": END,
        "not useful": "websearch",
        "not grounded/supported": "generate",
    }
)

app = workflow.compile()

# Test the workflow
inputs = {
    "question": "Is TensorFlow created by Google?",
}

for output in app.stream(inputs):
    for k, v in output.items():
        print(f"Finished running: {k}")
print("*****************************", v["answer"])
```

## Running the System

To run the system, execute the cells in the `local_agentic_rag.ipynb` notebook sequentially. Make sure your local LLM is running and your PDFs are in the `data` folder.

## Conclusion

This repository provides a comprehensive setup for an Agentic RAG system using a local LLM. It demonstrates the integration of LangChain, Chroma, and LangGraph to build a powerful document retrieval and question-answering system. Feel free to customize the workflow and experiment with different LLMs and document sources.
```

Feel free to modify any part of the README to better suit your needs.
