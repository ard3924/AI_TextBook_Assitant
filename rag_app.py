import os
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
import time # Import time for sleep
import streamlit as st # Import streamlit for caching
from googlesearch import search

# --- Configuration ---
load_dotenv()
INDEX_PATH = "faiss_index/textbook.index"
METADATA_PATH = "faiss_index/textbook.metadata"
GEMINI_MODEL = "gemini-2.5-pro" # Using available model for free tier

# --- System Prompt for Gemini ---
SYS_PROMPT = """You are an engaging academic assistant. Answer questions based *only* on the provided textbook context, keeping answers concise, interesting, and accessible. Do not cite sources or use external knowledge.

After the main answer, suggest 1-2 related follow-up questions for deeper understanding, with brief answers.

If you find a relevant answer in the context, end your response by asking: "Would you like me to search external websites for additional information on this topic?"

If the context doesn't provide an answer, ask: "I couldn't find this information in the textbook. Would you like me to search external websites for relevant answers?"
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

def ask_gemini(formatted_context, query, book_title, stream=False):
    """Sends the context and query to the cached Gemini model."""
    gemini_model = get_gemini_model()
    if gemini_model is None:
        if stream:
            for word in "Error: Gemini model could not be initialized. Please check your API key.".split():
                yield word + " "
                time.sleep(0.05)  # Small delay for lively effect
            return
        return "Error: Gemini model could not be initialized. Please check your API key."

    max_retries = 5
    base_delay = 2 # seconds
    prompt = f"{formatted_context}\n\nQUESTION: {query}\n\nBOOK_TITLE: {book_title}"


    for attempt in range(max_retries):
        try:
            # Use streaming for a more dynamic user experience
            response = gemini_model.generate_content(prompt, stream=stream)
            if stream:
                for chunk in response:
                    if chunk.text:
                        words = chunk.text.split()
                        for word in words:
                            yield word + " "
                            time.sleep(0.05)  # Small delay for smooth, lively flow
                return
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
                if stream:
                    for word in "Error: Gemini API call failed.".split():
                        yield word + " "
                        time.sleep(0.05)
                    return
                return "Error: Gemini API call failed."

    if stream:
        for word in "Error: Failed to get a response from Gemini after multiple retries due to rate limits.".split():
            yield word + " "
            time.sleep(0.05)
        return
    return "Error: Failed to get a response from Gemini after multiple retries due to rate limits."

def search_external_sources(query, num_results=5):
    """Searches external websites for additional information."""
    try:
        results = list(search(query, num_results=num_results))
        return results
    except Exception as e:
        print(f"Error during external search: {e}")
        return []

def generate_answer_from_external(query, external_results):
    """Generates an answer using Gemini based on external search results."""
    if not external_results:
        return "No external information could be retrieved."

    # Format external results for Gemini
    external_context = "EXTERNAL SEARCH RESULTS:\n"
    for i, url in enumerate(external_results, 1):
        external_context += f"{i}. {url}\n"

    prompt = f"""Based on the following external search results, provide a comprehensive answer to the question: "{query}"

{external_context}

Please provide a detailed answer based on these external sources. Include relevant information and cite the sources where possible."""

    # Use gemini-2.5-pro for external search
    try:
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        external_model = genai.GenerativeModel("gemini-2.5-pro", system_instruction="You are a helpful assistant that provides comprehensive answers based on external search results.")
        response = external_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating answer from external sources: {e}")
        return f"Error generating answer from external sources: {e}"

def query_rag(query, book_title, top_k, similarity_threshold, embed_model, stream=False):
    """Main RAG pipeline function."""

    # 0. Load FAISS index and metadata from cache
    index, metadata = load_rag_components()
    if index is None or metadata is None:
        if stream:
            yield "Error: The textbook index is not loaded. Please ingest the textbook first."
            return
        return "Error: The textbook index is not loaded. Please ingest the textbook first.", "", []

    # 1. Embed query (Locally, for free)
    query_vec = embed_query(query, embed_model)
    if query_vec is None:
        if stream:
            yield "Error: Could not embed your query."
            return
        return "Error: Could not embed your query.", "", []

    # 2. Retrieve chunks (Locally, for free)
    retrieved_chunks = retrieve_chunks(query_vec, index, metadata, top_k, similarity_threshold)

    # 3. Format context (Locally, for free)
    formatted_context = format_context(retrieved_chunks)

    # 4. Generate answer (Uses Gemini API free tier)
    if stream:
        answer_generator = ask_gemini(formatted_context, query, book_title, stream=True)
        full_answer = ""
        for chunk in answer_generator:
            full_answer += chunk
            yield chunk
        answer = full_answer
    else:
        answer = ask_gemini(formatted_context, query, book_title, stream=False)

    # 5. Check if user wants external search
    if "Would you like me to search external websites" in answer:
        # Extract the permission question and handle it in UI
        if stream:
            yield "\n\n"  # Add spacing for sources
        return answer, "", retrieved_chunks, True  # Add flag for external search request

    # 6. Final check
    if "INSUFFICIENT_KB" in answer:
        if stream:
            yield "\n\n"  # Add spacing for sources
        return "I could not find a relevant answer in the provided textbook.", "", []

    # 7. Format sources for display
    formatted_sources = format_sources(retrieved_chunks, book_title)

    if stream:
        yield "\n\n" + formatted_sources  # Yield sources after answer

    return answer, formatted_sources, retrieved_chunks, False
