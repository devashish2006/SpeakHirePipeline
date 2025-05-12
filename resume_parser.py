import fitz  # PyMuPDF
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant  # Updated import
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(file_path):
    """Extract text from PDF"""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def initialize_qdrant():
    """Initialize Qdrant collection"""
    client = QdrantClient("localhost", port=6333)
    
    # Delete old collection (if needed)
    try:
        client.delete_collection("resumes")
    except Exception:
        pass
    
    # Create new collection
    client.create_collection(
        collection_name="resumes",
        vectors_config=qdrant_models.VectorParams(
            size=768,  # Gemini embeddings dimension
            distance=qdrant_models.Distance.COSINE,
        )
    )
    return client

def process_resume(file_path, candidate_id):
    """Process and store resume in Qdrant"""
    # Extract text
    text = extract_text_from_pdf(file_path)
    
    # Split into chunks (to avoid exceeding embedding limits)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Initialize Qdrant
    client = initialize_qdrant()
    
    # Store chunks in Qdrant with metadata
    Qdrant.from_texts(
        texts=chunks,
        embedding=embeddings,
        url="http://localhost:6333",
        collection_name="resumes",
        metadatas=[{"candidate_id": candidate_id, "source": file_path} for _ in chunks],
    )
    print(f"✅ Resume stored for {candidate_id}")

if __name__ == "__main__":
    process_resume(
        file_path=r"E:\GenAi_FastPace\Ai_Interview\resume_samples\candidate1.pdf",
        candidate_id="candidate_1"
    )