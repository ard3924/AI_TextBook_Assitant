import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
import time # Import time for sleep
import streamlit as st # Import streamlit for caching

# --- Configuration ---
load_dotenv()
INDEX_PATH = "faiss_index/textbook.index"
METADATA_PATH = "faiss_index/textbook.metadata"
GEMINI_MODEL = "gemini-2.5-pro" # Using available model for free tier

# --- System Prompt for Gemini ---
SYS_PROMPT = """You are an advanced academic assistant. Your task is to answer questions based *only* on the provided textbook context.
You must ground your entire answer in the provided context.
Provide a comprehensive, detailed explanation in your answer, including examples where relevant, and break down complex concepts step-by-step. Do not cite sources or mention page numbers. Do not use any external knowledge.

After providing the main answer, autonomously generate and answer 2-3 related follow-up questions that deepen understanding of the topic, based solely on the provided context. Format these as:
**Follow-up Questions:**
1. [Question 1]
   [Detailed answer to question 1]
2. [Question 2]
   [Detailed answer to question 2]
3. [Question 3]
   [Detailed answer to question 3]

If the answer is not found in the provided context, or if the context is insufficient to answer the question, you must reply *only* with the single string:
INSUFFICIENT_KB
"""

@st.cache_resource
def get_gemini_model():
    """Initializes and caches the Gemini Generative Model client."""
    try:
        print("Initializing Gemini model... (This will only happen once)")
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYS_PROMPT)
        return model
    except Exception as e:
        print(f"Error initializing Google API clients: {e}")
        return None

@st.cache_resource
def load_rag_components():
    """Loads and caches the FAISS index and metadata."""
    try:
        print("Loading FAISS index and metadata...")
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        print("FAISS components loaded successfully.")
        return index, metadata
    except Exception as e:
        print(f"Error loading FAISS components: {e}")
        return None, None

def embed_query(query, embed_model):
    """Generates and normalizes embedding for the user query using the local model."""
    try:
        # Use the local SentenceTransformer model
        embedding = embed_model.encode([query], convert_to_numpy=True)
        embedding = embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(embedding) # Normalize for cosine similarity
        return embedding
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None

def retrieve_chunks(query_embedding, index, metadata, top_k, similarity_threshold):
    """Retrieves relevant chunks from the FAISS index."""
    if index is None or metadata is None:
        print("Error: Index or metadata not loaded.")
        return []

    D, I = index.search(query_embedding, top_k)
    results = []
    for i in range(min(top_k, len(I[0]))): # Use top_k parameter
        idx = I[0][i]
        similarity = D[0][i]
        
        if similarity >= similarity_threshold: # Use similarity_threshold parameter
            results.append(metadata[idx])
            
    return results

def format_context(retrieved_chunks):
    """Formats retrieved chunks into a string for the LLM prompt."""
    if not retrieved_chunks:
        return "No relevant context found."
    
    context_str = "CONTEXT:\n---\n"
    for chunk in retrieved_chunks:
        context_str += f"[Source: Page {chunk['page']}]\n"
        context_str += chunk['text']
        context_str += "\n---\n"
    return context_str

def format_sources(retrieved_chunks, book_title):
    """Formats the sources from retrieved chunks for display."""
    if not retrieved_chunks:
        return ""

    # Using a set to avoid duplicate page numbers
    pages = sorted(list(set(chunk['page'] for chunk in retrieved_chunks)))

    # For now, we only have page numbers. A more advanced implementation could infer chapter/section.
    source_str = f'\n\n**Sources:**\n- "{book_title}" — Pages: {", ".join(map(str, pages))}'
    return source_str

def ask_gemini(formatted_context, query, book_title):
    """Sends the context and query to the cached Gemini model."""
    gemini_model = get_gemini_model()
    if gemini_model is None:
        return "Error: Gemini model could not be initialized. Please check your API key."

    max_retries = 5
    base_delay = 2 # seconds
    prompt = f"{formatted_context}\n\nQUESTION: {query}\n\nBOOK_TITLE: {book_title}"
    
    
    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_str = str(e)
            # Retry on 429 (rate limit) and 500 (internal server error)
            if ("429" in error_str or "500" in error_str) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"API Error detected. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"Error calling Gemini API: {e}")
                return "Error: Gemini API call failed."
    
    return "Error: Failed to get a response from Gemini after multiple retries due to rate limits."

def query_rag(query, book_title, top_k, similarity_threshold, embed_model):
    """Main RAG pipeline function."""

    # 0. Load FAISS index and metadata from cache
    index, metadata = load_rag_components()
    if index is None or metadata is None:
        return "Error: The textbook index is not loaded. Please ingest the textbook first.", "", []

    # 1. Embed query (Locally, for free)
    query_vec = embed_query(query, embed_model)
    if query_vec is None:
        return "Error: Could not embed your query.", "", []

    # 2. Retrieve chunks (Locally, for free)
    retrieved_chunks = retrieve_chunks(query_vec, index, metadata, top_k, similarity_threshold)
    if not retrieved_chunks:
        return "I could not find a relevant answer in the provided textbook.", "", []

    # 3. Format context (Locally, for free)
    formatted_context = format_context(retrieved_chunks)

    # 4. Generate answer (Uses Gemini API free tier)
    answer = ask_gemini(formatted_context, query, book_title)

    # 5. Final check
    if "INSUFFICIENT_KB" in answer:
        return "I could not find a relevant answer in the provided textbook.", "", []

    # 6. Format sources for display
    formatted_sources = format_sources(retrieved_chunks, book_title)

    return answer, formatted_sources, retrieved_chunks
