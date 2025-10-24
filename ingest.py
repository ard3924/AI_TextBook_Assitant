import os
import pickle
import numpy as np
import faiss
import pdfplumber
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
PDF_PATH = "data/textbook.pdf"
INDEX_PATH = "faiss_index/textbook.index"
METADATA_PATH = "faiss_index/textbook.metadata"
CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 50

# Ensure directories exist
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)

def extract_text_from_pdf():
    """Extracts text and page numbers from the PDF."""
    print(f"Starting PDF extraction from {PDF_PATH}...")
    page_texts = []
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}")
        return []
        
    with pdfplumber.open(PDF_PATH) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                page_texts.append({"page": i + 1, "text": text})
    print(f"Extracted {len(page_texts)} pages.")
    return page_texts

def chunk_text(page_texts):
    """Chunks text into overlapping word-based chunks with metadata."""
    print("Starting text chunking...")
    chunks = []
    chunk_id = 0
    
    for page_data in page_texts:
        text = page_data["text"]
        page_num = page_data["page"]
        words = text.split()
        
        for i in range(0, len(words), CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS):
            chunk_words = words[i : i + CHUNK_SIZE_WORDS]
            if not chunk_words:
                continue
                
            chunk_text = " ".join(chunk_words)
            metadata = {
                "page": page_num,
                "chunk_id": f"page_{page_num}_chunk_{chunk_id}",
                "text": chunk_text
            }
            chunks.append(metadata)
            chunk_id += 1
            
    print(f"Created {len(chunks)} text chunks.")
    return chunks

def get_embeddings(texts, embed_model):
    """Generates embeddings for a list of texts using the local model."""
    print(f"Generating local embeddings for {len(texts)} chunks...")
    try:
        # Use the local SentenceTransformer model
        embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.astype('float32')
    except Exception as e:
        print(f"Error during local embedding generation: {e}")
        return None

def build_and_save_faiss_index(embeddings, metadata, embed_model):
    """Builds, normalizes, and saves a FAISS index and metadata."""
    if embeddings is None:
        print("No embeddings to process.")
        return

    print("Building FAISS index...")
    # Get dimension from the model itself
    d = embed_model.get_sentence_embedding_dimension()
    
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}")
    
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to {METADATA_PATH}")

def main(embed_model):
    """Main ingestion pipeline."""
    print("--- Starting Ingestion Pipeline ---")
    page_texts = extract_text_from_pdf()
    if not page_texts:
        print("Ingestion failed: Could not extract text from PDF.")
        return

    chunks_with_metadata = chunk_text(page_texts)
    if not chunks_with_metadata:
        print("Ingestion failed: Could not create text chunks.")
        return

    texts_to_embed = [chunk['text'] for chunk in chunks_with_metadata]
    embeddings = get_embeddings(texts_to_embed, embed_model)
    
    if embeddings is not None:
        build_and_save_faiss_index(embeddings, chunks_with_metadata, embed_model)
        print("--- Ingestion Pipeline Complete ---")
    else:
        print("Ingestion failed: Could not generate embeddings.")

if __name__ == "__main__":
    # This allows running ingest.py directly for testing/debugging
    from model_loader import get_embedding_model
    model = get_embedding_model()
    main(model)