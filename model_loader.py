# model_loader.py
import streamlit as st
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'

@st.cache_resource
def get_embedding_model():
    """
    Loads the SentenceTransformer model and caches it using Streamlit.
    This function will only be run once, and the returned model will be reused
    on subsequent page loads.
    """
    print("Loading local embedding model... (This will only happen once)")
    return SentenceTransformer(EMBED_MODEL_NAME)