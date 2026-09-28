import os
import re
import glob

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()  

DATA_DIR = "data"
DB_DIR = "chroma_store"


# 1. LOAD ---- read text files from data
def load_documents():

    docs = []
    
    for path in glob.glob(f"{DATA_DIR}/*.txt"):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        filename = os.path.basename(path)
        doc = Document(
            page_content=content,
            metadata={"source_type": "text", "source": filename}
        )
        docs.append(doc)

    return docs


# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-nemotron-embed-vl-1b-v2",
        truncate="NONE",
        )
    # embeddings = GoogleGenerativeAIEmbeddings(
    #     model="gemini-embedding-001"
    # )
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    docs = load_documents()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
    ).split_documents(docs)

    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 6})


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("who is the king of westeros?")
    
    for r in results:
        print(f"[{r.metadata.get('source_type', 'unknown').upper()}: {r.metadata.get('source', 'unknown')}] {r.page_content[:150]}...\n")

    
