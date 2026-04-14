import os
import numpy as np
import gradio as gr
import torch
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

k=3

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/gemma-2b-it"



import os
from huggingface_hub import login
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN is not set. Please add it in the Space settings.")
login(token=hf_token)



print("Loading models... this may take a moment.")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    device_map="auto",
)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md", ".csv"}


def normalize_file_path(uploaded_file):
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, str):
        return uploaded_file
    if hasattr(uploaded_file, "name"):
        return uploaded_file.name
    if isinstance(uploaded_file, dict):
        return uploaded_file.get("name")
    return None


def load_document(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    return loader.load()


def build_prompt(context, question):
    return f"""
You are a precise assistant.
Answer the question in EXACTLY two sentences.
Do not use bullet points or numbering.
Context:
{context}
Question: {question}
Answer:
""".strip()


def answer_question(uploaded_file, question):
    file_path = normalize_file_path(uploaded_file)
    if not file_path:
        return "Please upload a document file."

    if not question or not question.strip():
        return "Please type a question to ask about the uploaded document."

    document = load_document(file_path)
    if not document:
        return (
            "Could not read the document. Upload a supported file type: .txt, .pdf, .docx, .md, or .csv."
        )

    chunks=text_splitter.split_documents(document)
    if not question:
        return ""

    
    db = Chroma.from_documents(chunks, embedding=embedding_model)
    retriever = db.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)


    if not relevant_docs:
        return "No relevant information found in the document."

    prompt = build_prompt(context,question)
    output = generator(
        prompt,
        max_new_tokens=200,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )
    return output[0].get("generated_text", "").strip()


def summarize_document(uploaded_file):
    file_path = normalize_file_path(uploaded_file)
    if not file_path:
        return "Please upload a document file."

    text = load_document(file_path)
    if not text:
        return (
            "Could not read the document. Upload a supported file type: .txt, .pdf, .docx, .md, or .csv."
        )

    prompt = f"""
You are a concise assistant.
Summarize the following document in exactly two sentences.
Document:
{text}
Summary:
""".strip()
    output = generator(
        prompt,
        max_new_tokens=200,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
    )
    return output[0].get("generated_text", "").strip()


def create_ui():
    with gr.Blocks(title="Document QA + Summarization") as demo:
        gr.Markdown(
            """
# Document Q&A
Upload a document and ask a question about its content.
Supported file types: `.txt`, `.pdf`, `.docx`, `.md`, `.csv`.
"""
        )

        with gr.Row():
            file_input = gr.File(label="Upload document", file_types=[".txt", ".pdf", ".docx", ".md", ".csv"])

        question_input = gr.Textbox(
            label="Question",
            placeholder="Ask something about the uploaded document...",
            lines=2,
        )
        answer_output = gr.Textbox(label="Answer", lines=6)
        summary_output = gr.Textbox(label="Document Summary", lines=4)

        question_button = gr.Button("Answer question")
        summary_button = gr.Button("Summarize document")

        question_button.click(answer_question, inputs=[file_input, question_input], outputs=answer_output)
        summary_button.click(summarize_document, inputs=[file_input], outputs=summary_output)

    return demo


demo = create_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))