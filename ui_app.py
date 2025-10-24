import streamlit as st
import os
import rag_app  # Imports the core logic
from ingest import main as run_ingest # Imports the ingest function
from model_loader import get_embedding_model # Import the cached model loader

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Textbook Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Textbook Assistant MVP")
st.caption("Powered by Google Gemini & FAISS")

# --- Load Shared Model ---
embed_model = get_embedding_model()

# --- Initialize Session State ---
if 'ingested' not in st.session_state:
    st.session_state.ingested = False
if 'book_title' not in st.session_state:
    st.session_state.book_title = "My Textbook"
# We will also use session_state to store the chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for Setup ---
with st.sidebar:
    st.header("Setup & Ingestion")
    
    uploaded_file = st.file_uploader("1. Upload Textbook (PDF)", type=["pdf"])
    
    if uploaded_file:
        st.session_state.book_title = uploaded_file.name.replace('.pdf', '')
        
        if st.button("2. Process & Ingest Textbook"):
            data_dir = "data"
            pdf_path = os.path.join(data_dir, "textbook.pdf")
            os.makedirs(data_dir, exist_ok=True)
            
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info(f"File '{uploaded_file.name}' saved. Starting ingestion...")
            
            with st.spinner("Processing PDF, chunking, and embedding... This may take a few minutes."):
                try:
                    run_ingest(embed_model) # Pass the loaded model
                    st.session_state.ingested = True
                    st.success("Textbook ingested successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                    st.session_state.ingested = False
    
    st.divider()

    # Define default values for RAG parameters (optimized for maximum free tier output)
    DEFAULT_TOP_K = 10
    DEFAULT_SIMILARITY_THRESHOLD = 0.3

    st.header("RAG Parameters")
    # Allow users to adjust TOP_K and SIMILARITY_THRESHOLD
    # Use st.session_state to persist these values
    st.session_state.top_k = st.slider(
        "Number of chunks to retrieve (TOP_K)", min_value=1, max_value=10, value=DEFAULT_TOP_K, step=1
    )
    st.session_state.similarity_threshold = st.slider(
        "Similarity Threshold", min_value=0.0, max_value=1.0, value=DEFAULT_SIMILARITY_THRESHOLD, step=0.05
    )
    st.caption("Adjust these to fine-tune retrieval. Lower threshold or higher TOP_K might retrieve more context.")
    
    # Add a button to clear the chat history
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun() # Rerun the app to reflect the cleared state

    if st.session_state.ingested or os.path.exists("faiss_index/textbook.index"):
        st.success(f"Status: Ready to query '{st.session_state.book_title}'!")
    else:
        st.warning("Status: No textbook ingested. Please upload and ingest a PDF.")

# --- Main Chat Interface ---
st.header("Ask a Question")
st.info("This assistant is running 100% on Google Gemini.")

# Display past messages from session_state
for message in st.session_state.messages:
    role = message["role"]
    with st.chat_message(role):
        if role == "user":
            st.markdown(message["content"])
        else: # Assistant message
            st.markdown(message["answer"])
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    st.markdown(message["sources"].strip())


# --- NEW CHAT INPUT LOGIC ---
# Use st.chat_input which is designed for this purpose.
# It runs *after* the rest of the script.
if query := st.chat_input(f"Ask a question about {st.session_state.book_title}..."):
    
    # 1. Check for ingestion
    if not (st.session_state.ingested or os.path.exists("faiss_index/textbook.index")):
        st.error("Please upload and ingest a textbook using the sidebar first.")
    else:
        # 2. Add user's query to session state and display it
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # 3. Get the answer from the RAG pipeline
        with st.spinner("Searching textbook and generating answer..."):
            try:
                answer, sources, retrieved_chunks = rag_app.query_rag(
                    query, 
                    st.session_state.book_title, # Pass book_title
                    st.session_state.top_k, # Pass TOP_K from session state
                    st.session_state.similarity_threshold, # Pass SIMILARITY_THRESHOLD from session state
                    embed_model # Pass the loaded model
                )
                
                # 4. Store answer and sources separately in session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "answer": answer, 
                    "sources": sources
                })
                
                # 5. Display the response in a more structured way
                with st.chat_message("assistant"): # This block displays the *newest* message
                    st.markdown(answer) # Display the main answer
                    
                    if sources: # If there are sources, show them in an expander
                        with st.expander("View Sources"):
                            st.markdown(sources.strip())
                    
                    # Show the raw retrieved context for debugging
                    with st.expander("Show Retrieved Context"):
                        st.json(retrieved_chunks)

            except Exception as e:
                st.error(f"An error occurred during query processing: {e}")