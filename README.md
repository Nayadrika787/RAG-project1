# 📄 Document Q&A and Summarization using Gemma 2B

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to upload documents and interact with them through **context-aware question answering** and **concise summarization**. The project is deployed on **Hugging Face Spaces** with an intuitive **Gradio** interface.

HF space link: https://huggingface.co/spaces/nayadrika/1rag_qa
---

## Features

- **Document Upload** – Supports `.txt`, `.pdf`, `.docx`, `.md`, and `.csv` files.
- **Context-Aware Question Answering** – Uses RAG to provide accurate answers based on document content.
- **Document Summarization** – Generates concise summaries capturing key concepts.
- **Semantic Search** – Employs Utilizes the `sentence-transformers/all-MiniLM-L6-v2` embedding model from Hugging Face to enable efficient document retrieval.
- **Interactive UI** – Built with Gradio for an easy-to-use experience.
- **Cloud Deployment** – Hosted on Hugging Face Spaces.
- **Secure Token Handling** – Utilizes Hugging Face secrets for authentication.

---

## System Architecture

### Question Answering (RAG)
The Question Answering module uses a Retrieval-Augmented Generation (RAG) pipeline:
1. Upload  document
2. Load  document using different loaders according to its type
3. Split the document text
4. Using Sentence Transformers, embeddings are generated for the chunks
5. Storage in Chroma Vector Database
6. Retrieval of relevant context using a Retriever
7. Prompt construction with context and user question
8. Answer generation using the Gemma 2B model

### 📝 Document Summarization
The Summarization module follows a simpler pipeline:
1. Upload  document
2. Load  document using different loaders according to its type
3. Prompt construction with the full document
4. Summary generation using the Gemma 2B model
