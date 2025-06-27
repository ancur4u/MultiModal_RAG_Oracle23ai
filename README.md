🤖 MultiModal RAG Assistant using Oracle 23ai

A Retrieval-Augmented Generation (RAG) assistant that lets you chat with your documents using OpenAI + Oracle Database 23ai. Supports multi-modal document ingestion, secure credential setup, vector-based chunk retrieval, and a fully mobile-responsive Streamlit UI.

🔍 Features
	•	📄 Ingest multi-modal files: PDF, DOCX, PPTX, TXT, Excel, Images (OCR), Markdown, HTML
	•	🧠 Chunk & embed with SentenceTransformers and OpenAI (or offline if needed)
	•	💬 Query with GPT and get contextual answers with source citations
	•	🗃️ Uses Oracle 23ai VECTOR datatype for fast vector search
	•	📈 Visual analytics: file stats, chunk counts, and usage trends
	•	🔐 Secure, session-based credential setup (OpenAI API & Oracle DB wallet)
	•	📱 Mobile-first responsive UI with auto-layout adaptation

📦 Supported File Types

<img width="643" alt="image" src="https://github.com/user-attachments/assets/8fc2567a-7f73-433b-827c-a683b6e29198" />

🚀 Getting Started

1. Clone the repo

   git clone https://github.com/ancur4u/MultiModal_RAG_Oracle23ai.git
   
   cd MultiModal_RAG_Oracle23ai
   
2. Create virtual environment
   
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   
3. Install dependencies
   
   pip install -r requirements.txt
4. Configure .env (optional)
   
   OPENAI_API_KEY=sk-...
   ORACLE_USERNAME=your_username
   ORACLE_PASSWORD=your_password
   ORACLE_SERVICE_NAME=your_service_high
   ORACLE_WALLET_PATH=/path/to/wallet
   
5. Launch the App.
   streamlit run streamlit run RAG_Oracle23ai_Final.py

🧠 System Architecture

📄 Documents → 🧠 Chunking + Embeddings → 📊 Oracle VECTOR Search
                                      ↘
                                🤖 GPT-based Q&A
                                
	•	Chunks are embedded using all-MiniLM-L6-v2 or OpenAI API
  •	Vectors stored in Oracle 23ai (VECTOR(384, FLOAT32))
	•	GPT (via OpenAI) used for context-based answers

📊 Analytics & Admin Tools

	•	📚 Document summary: file type, chunk count, ingestion date
	•	💬 Chat history viewer with timestamps
	•	📈 Config panel: model, top-k, chunk size
	•	✅ Credential verifier for OpenAI & Oracle

🛠️ Planned Enhancements

	•	Offline GPT via Ollama
	•	Audio input + Whisper integration
	•	PDF summarization mode
	•	Docker containerization
	•	Multi-user login & access control


