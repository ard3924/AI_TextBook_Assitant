# 📚 AI Textbook Assistant

This project is a sophisticated Retrieval-Augmented Generation (RAG) application that transforms any PDF textbook into an interactive, intelligent academic assistant. Users can ask complex questions and receive detailed, context-aware answers based exclusively on the content of the provided textbook.

The assistant is powered by Google's Gemini Pro model for generation and a local Sentence-Transformer model for efficient, free-of-charge embeddings.

## ✨ Features

- **PDF Ingestion**: Upload any PDF textbook to serve as the knowledge base.
- **Contextual Q&A**: Get answers that are strictly grounded in the textbook's content, preventing hallucinations or external information.
- **Source Citing**: Every answer is accompanied by the specific page numbers from the textbook where the information was found.
- **Proactive Learning**: The AI autonomously generates and answers relevant follow-up questions to deepen the user's understanding of the topic.
- **Configurable Retrieval**: Fine-tune the retrieval process by adjusting the number of chunks (`TOP_K`) and the `Similarity Threshold` directly in the UI.
- **Stateful Chat Interface**: A clean, modern chat interface that remembers your conversation history.
- **Efficient & Local**: Uses a local embedding model and a FAISS vector store for fast, private, and cost-effective retrieval. All resource-intensive models are cached for performance.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Generative LLM**: Google Gemini (`gemini-2.5-pro`)
- **Embedding Model**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **PDF Parsing**: pdfplumber

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
    streamlit run ui_app.py
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

---

This README provides a comprehensive guide to your project. Let me know if you'd like any section expanded or modified!