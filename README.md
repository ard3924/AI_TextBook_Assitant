# 📚 AI Textbook Assistant

This project is a sophisticated Retrieval-Augmented Generation (RAG) application that transforms any PDF textbook into an interactive, intelligent academic assistant. Users can ask complex questions and receive detailed, context-aware answers based exclusively on the content of the provided textbook.

The assistant is powered by Google's Gemini 2.5 Pro model for generation and a local Sentence-Transformer model for efficient, free-of-charge embeddings.

## ✨ Features

- **PDF Ingestion**: Upload any PDF textbook to serve as the knowledge base.
- **Contextual Q&A**: Get answers that are strictly grounded in the textbook's content, preventing hallucinations or external information.
- **Source Citing**: Every answer is accompanied by the specific page numbers from the textbook where the information was found.
- **Proactive Learning**: The AI autonomously generates and answers relevant follow-up questions to deepen the user's understanding of the topic.
- **External Search Integration**: When textbook knowledge is insufficient or users want additional information, the AI can search external websites and generate comprehensive answers using Gemini 2.5 Pro.
- **Configurable Retrieval**: Fine-tune the retrieval process by adjusting the number of retrieved chunks (`TOP_K`) and the `Similarity Threshold` directly in the UI.
- **Stateful Chat Interface**: A clean, modern chat interface that remembers your conversation history.
- **Efficient & Local**: Uses a local embedding model and a FAISS vector store for fast, private, and cost-effective retrieval. All resource-intensive models are cached for performance.

## 🛠️ Tech Stack

- **Frontend**: Streamlit (`1.35.0`)
- **Backend**: Python
- **Generative LLM**: Google Gemini (`gemini-2.5-pro` via `google-generativeai==0.5.4`)
- **Embedding Model**: Sentence-Transformers (`2.7.0`)
- **Vector Database**: FAISS (`faiss-cpu==1.8.0`)
- **PDF Parsing**: pdfplumber (`0.11.1`)
- **Environment Management**: python-dotenv (`1.0.1`)

## 📂 Project Structure
```
backend/
├── data/
│   └── textbook.pdf        # Your ingested PDF goes here
├── faiss_index/
│   ├── textbook.index      # The FAISS vector index
│   └── textbook.metadata   # The metadata for the text chunks
├── .env                    # Stores API keys
├── ingest.py               # Handles PDF parsing, chunking, and vectorization
├── model_loader.py         # Caches and serves the embedding model
├── rag_app.py              # Core RAG logic and communication with Gemini
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── ui_app.py               # The main Streamlit application file
```

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

- Python 3.8+
- A Google API Key with the Gemini API enabled. You can get one from Google AI Studio.

### 2. Setup

1.  **Clone the repository (or set up the project folder):**
    Navigate to your project's `backend` directory.

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    ```
    Activate it:
    -   On Windows: `.\venv\Scripts\activate`
    -   On macOS/Linux: `source venv/bin/activate`

3.  **Install Dependencies:**
    Install all required Python packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a file named `.env` in the `backend` directory and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```

### 3. Running the Application

1.  **Launch the Streamlit App:**
    Run the following command in your terminal from the `backend` directory:
    ```bash
    py -m streamlit run ui_app.py
    ```
2.  **Open in Browser:**
    The application will automatically open in your default web browser.

## 📖 How to Use

1.  **Upload Textbook**: In the sidebar, use the file uploader to select the PDF textbook you want to use.
2.  **Ingest the Document**: Click the **"Process & Ingest Textbook"** button. This will:
    -   Extract text from the PDF.
    -   Split the text into smaller, overlapping chunks.
    -   Generate embeddings for each chunk using the local model.
    -   Build and save a FAISS vector index for fast retrieval.
    This process may take a few minutes, depending on the size of the PDF.
3.  **Ask Questions**: Once ingestion is complete, the status in the sidebar will turn to "Ready". You can now type your questions into the chat input at the bottom of the main screen.
4.  **Adjust Parameters (Optional)**: Use the sliders in the sidebar to adjust the `TOP_K` and `Similarity Threshold` to fine-tune the retrieval process for better answers.
5.  **Clear Conversation**: If you want to start a new chat session, click the "Clear Conversation" button in the sidebar.
6.  **External Search (Optional)**: If the AI suggests searching external websites for additional information, click "Yes, search external websites" to get comprehensive answers from web sources.

## ⚠️ Limitations

This section outlines the current limitations of the application.

### Technical Limitations
- **PDF Quality Dependency**: The accuracy of answers heavily depends on the quality of text extraction from the source PDF. Scanned documents (non-selectable text) or PDFs with complex multi-column layouts may result in poor extraction and inaccurate context.
- **Chunking Strategy**: Text is currently chunked into fixed-size word segments. This simple method can occasionally split a sentence or a coherent thought across two different chunks, potentially reducing the quality of the retrieved context.
- **Embedding Model Scope**: The application uses `all-MiniLM-L6-v2`, which is highly efficient but may not capture the nuances of highly specialized or niche academic domains as effectively as larger, more specialized embedding models.
- **Vector Search Approximation**: FAISS utilizes Approximate Nearest Neighbor (ANN) search to ensure high speed. While highly accurate, it is not exhaustive and may, in rare cases, miss the single most relevant text chunk.

### API & Service Limitations
- **Gemini API Quota**: The application relies on the Google Gemini API, which has rate limits (e.g., requests per minute) and usage quotas, especially on the free tier. The app includes a retry mechanism, but heavy usage can lead to temporary service denial.
- **Model Context Window**: Gemini 2.5 Pro has a context limit that may constrain the amount of retrieved text that can be processed in a single query.
- **External Search Reliability**: The external search feature uses an unofficial Google Search library. This can be unreliable and may be affected by network restrictions or changes in Google's web structure.

### Functional Limitations
- **Single Textbook Focus**: The current architecture is designed to process and query a single textbook at a time. Ingesting a new PDF will overwrite the previous index.
- **Text-Only Processing**: The ingestion pipeline extracts and processes **text only**. It cannot interpret or analyze images, tables, complex mathematical formulas, or diagrams within the PDF.
- **No Persistent Memory**: The chat history is stored in the user's session state and will be lost if the browser tab is closed or the application is restarted.
- **English-Centric**: The embedding model and system prompts are optimized for English. Performance with non-English languages will be significantly degraded.

### Performance & Resource Considerations
- **Initial Ingestion Time**: The first-time ingestion of a large PDF can be time-consuming and CPU-intensive, as it involves chunking and generating embeddings for the entire document.
- **Memory (RAM) Usage**: Loading the embedding model and processing large documents can require a significant amount of system RAM.
- **Storage Footprint**: The generated FAISS index and metadata files can consume considerable disk space, with their size scaling directly with the length of the source PDF.

### Security & Privacy
- **API Key Management**: The Google API key is stored in a local `.env` file. It is crucial to ensure this file is never committed to public version control (e.g., a public GitHub repository).
- **External Search Privacy**: When the external search feature is used, the user's query is sent to Google's public search engine. Users should be mindful of this when querying sensitive topics.

---

This README provides a comprehensive guide to your project. Let me know if you'd like any section expanded or modified!
