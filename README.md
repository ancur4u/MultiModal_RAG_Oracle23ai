# 🤖 MultiModal RAG Assistant with Oracle 23ai

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![Oracle](https://img.shields.io/badge/Oracle-23ai-orange.svg)](https://oracle.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-green.svg)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A powerful, multi-modal Retrieval-Augmented Generation (RAG) application built with Oracle Database 23ai Vector Search, OpenAI GPT, and Streamlit. Features secure user authentication, document processing across multiple formats, and intelligent chat capabilities.

## 🎥 **Live Demo & Tutorial**

[![YouTube Demo](https://img.shields.io/badge/YouTube-Demo%20Video-red?logo=youtube)](https://youtu.be/FRPM9zABVVg)

**Watch the complete walkthrough:** [MultiModal RAG Assistant Demo](https://youtu.be/FRPM9zABVVg)

This screencast demonstrates:
- 🔐 User authentication and system setup
- 📄 Multi-format document upload and processing
- 🧠 AI-powered chat with document context
- 📱 Mobile-responsive interface
- ⚡ Oracle 23ai vector search in action

![MultiModal RAG Architecture](https://via.placeholder.com/800x400/667eea/ffffff?text=MultiModal+RAG+Assistant+Architecture)

## 🌟 Features

### 🔐 **Secure Authentication System**
- ✅ User registration and login with secure password hashing (PBKDF2)
- ✅ Session-based authentication with automatic expiration
- ✅ User data isolation and multi-tenant architecture
- ✅ Admin configuration interface for system setup

### 📄 **Multi-Modal Document Processing**
- ✅ **PDF Documents**: Text and table extraction with pdfplumber
- ✅ **Word Documents**: Full DOCX processing with python-docx
- ✅ **PowerPoint**: PPTX slide content extraction with python-pptx
- ✅ **Excel/CSV**: Spreadsheet processing with pandas
- ✅ **Images**: OCR text extraction with Pillow + pytesseract
- ✅ **Text Files**: Plain text and markdown support

### 🧠 **Advanced AI Capabilities**
- ✅ **Vector Search**: Oracle 23ai vector similarity search
- ✅ **Smart Chunking**: Intelligent content segmentation with sentence boundaries
- ✅ **Embedding Generation**: Sentence-transformers for semantic search
- ✅ **GPT Integration**: OpenAI GPT-3.5-turbo for contextual responses
- ✅ **Source Citations**: Automatic page and document references

### 📱 **Mobile-First Design**
- ✅ Responsive UI optimized for desktop and mobile devices
- ✅ Touch-friendly interface with intuitive navigation
- ✅ Progressive web app capabilities
- ✅ Adaptive chunking and search for mobile performance

### ⚡ **Enterprise-Ready**
- ✅ Oracle Database 23ai with vector search capabilities
- ✅ Scalable multi-user architecture
- ✅ Comprehensive error handling and logging
- ✅ Real-time processing with progress indicators

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Oracle Database 23ai instance with Vector Search enabled
- OpenAI API key
- Oracle Autonomous Database wallet (for cloud deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/ancur4u/MultiModal_RAG_Oracle23ai.git
cd MultiModal_RAG_Oracle23ai
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install streamlit sentence-transformers openai oracledb python-dotenv pdfplumber numpy pandas

# Install optional dependencies for full multi-modal support
pip install python-docx python-pptx openpyxl Pillow pytesseract beautifulsoup4 markdown

# For advanced table extraction (optional)
pip install camelot-py tabula-py
```

### 3. System Dependencies

**macOS:**
```bash
brew install tesseract poppler ghostscript
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils ghostscript
```

**Windows:**
- Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- Install [Poppler](https://github.com/oschwartz10612/poppler-windows)

### 4. Environment Configuration (Optional)

Create a `.env` file in the project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Oracle Database Configuration (optional - can be configured via UI)
ORACLE_USERNAME=admin
ORACLE_PASSWORD=your_db_password
ORACLE_SERVICE_NAME=your_service_name_high
ORACLE_WALLET_PATH=/path/to/wallet

# Security
SECRET_KEY=your_secret_key_here
```

### 5. Run the Application

```bash
streamlit run RAG_Oracle23ai_Final.py
```

The application will open in your browser at `http://localhost:8501`

## 🏗️ Application Architecture

### Authentication Flow
```
1. Landing Page (Login/Signup) 
   ↓
2. Admin Setup (System Configuration)
   ↓ 
3. Database Initialization
   ↓
4. Full RAG Application
```

### Data Flow
```
Document Upload → Multi-Modal Processing → Smart Chunking → 
Vector Embeddings → Oracle 23ai Storage → Vector Search → 
GPT Response Generation → User Interface
```

### Database Schema

#### Users Table
```sql
CREATE TABLE users (
    user_id VARCHAR2(50) PRIMARY KEY,
    username VARCHAR2(50) UNIQUE NOT NULL,
    email VARCHAR2(255),
    password_hash VARCHAR2(255) NOT NULL,
    salt VARCHAR2(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active NUMBER(1) DEFAULT 1
);
```

#### Documents Table
```sql
CREATE TABLE documents (
    document_id VARCHAR2(50) PRIMARY KEY,
    user_id VARCHAR2(50) NOT NULL,
    filename VARCHAR2(500) NOT NULL,
    file_type VARCHAR2(50) DEFAULT 'pdf',
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_chunks NUMBER DEFAULT 0,
    file_size NUMBER,
    CONSTRAINT fk_documents_user_id 
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
```

#### Document Chunks Table
```sql
CREATE TABLE document_chunks (
    chunk_id VARCHAR2(50) PRIMARY KEY,
    document_id VARCHAR2(50) NOT NULL,
    user_id VARCHAR2(50) NOT NULL,
    chunk_text CLOB NOT NULL,
    chunk_index NUMBER NOT NULL,
    page_number NUMBER DEFAULT 1,
    content_type VARCHAR2(50) DEFAULT 'text',
    metadata CLOB,
    embedding VECTOR(384, FLOAT32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🔧 Configuration Options

### Model Configuration
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Chunk Size**: 1000 characters (500 for mobile)
- **Overlap**: 200 characters
- **Top K Results**: 5 (3 for mobile)
- **Vector Distance**: COSINE similarity

### Security Configuration
- **Session Timeout**: 24 hours (configurable)
- **Password Requirements**: Minimum 6 characters (configurable)
- **Token Generation**: Cryptographically secure random tokens

## 📱 Usage Guide

### First-Time Setup (Admin)
1. Access the application URL
2. Click "Configure System" 
3. Enter OpenAI API key
4. Configure Oracle Database credentials
5. Test connections and initialize system

### User Registration
1. Click "Sign Up" on the landing page
2. Choose username and password
3. Optional: provide email address
4. Create account and sign in

### Document Processing
1. Upload supported file formats (PDF, DOCX, XLSX, etc.)
2. Wait for multi-modal processing to complete
3. View processing results and chunk statistics

### AI Chat
1. Ask questions about uploaded documents
2. Receive contextual responses with source citations
3. View similarity scores and page references
4. Clear chat history as needed

## 🧪 Testing

### Unit Tests
```bash
# Run basic functionality tests
python -m pytest tests/ -v
```

### Integration Tests
```bash
# Test database connectivity
python test_db_connection.py

# Test OpenAI integration
python test_openai_connection.py
```

### Load Testing
```bash
# Test with multiple concurrent users
locust -f locustfile.py --host=http://localhost:8501
```

## 🚀 Deployment

### Local Development
```bash
streamlit run RAG_Oracle23ai_Final.py --server.port 8501
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "RAG_Oracle23ai_Final.py", "--server.address", "0.0.0.0"]
```

### Cloud Deployment

#### Streamlit Cloud
1. Connect your GitHub repository
2. Set environment variables in Streamlit Cloud dashboard
3. Deploy directly from GitHub

#### Oracle Cloud Infrastructure
1. Create compute instance with Python 3.8+
2. Configure Oracle Autonomous Database
3. Set up reverse proxy with SSL/TLS

#### AWS/Azure/GCP
- Use container services (ECS, Container Instances, Cloud Run)
- Configure managed databases or use Oracle Cloud
- Set up load balancers and auto-scaling

## 🔍 Troubleshooting

### Common Issues

#### Database Connection Errors
```python
# Check TNS_ADMIN environment variable
echo $TNS_ADMIN

# Verify wallet files
ls -la /path/to/wallet/

# Test connection manually
python -c "import oracledb; print('Oracle client OK')"
```

#### OpenAI API Issues
```python
# Verify API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

#### Memory Issues with Large Documents
- Reduce chunk size in configuration
- Process documents in smaller batches
- Use streaming for large file uploads

#### Mobile Performance
- Ensure mobile-specific configurations are active
- Test on actual mobile devices
- Monitor network usage and optimize

### Error Logs
```bash
# View Streamlit logs
tail -f ~/.streamlit/logs/streamlit.log

# Application logs
tail -f app.log
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make changes and add tests
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all functions and classes
- Maintain test coverage above 80%

### Pull Request Process
1. Update README.md with details of changes
2. Update version numbers following semver
3. Ensure all tests pass
4. Request review from maintainers

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Oracle**: For Oracle Database 23ai and vector search capabilities
- **OpenAI**: For GPT models and embedding technologies
- **Streamlit**: For the amazing web application framework
- **Sentence Transformers**: For state-of-the-art embedding models
- **Community**: For testing, feedback, and contributions

## 📞 Support

- 📹 **Video Tutorial**: [YouTube Demo](https://youtu.be/FRPM9zABVVg)
- 🐛 **Issues**: [GitHub Issues](https://github.com/ancur4u/MultiModal_RAG_Oracle23ai/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ancur4u/MultiModal_RAG_Oracle23ai/discussions)
- 📧 **Email**: [Contact Author](mailto:ankr,prshr.ai@gmail.com)

## 🗺️ Roadmap

### Version 2.0 (Planned)
- [ ] Advanced OCR with layout detection
- [ ] Multi-language support
- [ ] Real-time collaboration features
- [ ] API endpoints for integration
- [ ] Advanced analytics dashboard

### Version 2.1 (Future)
- [ ] Voice input and output
- [ ] Integration with more LLM providers
- [ ] Advanced document versioning
- [ ] Export capabilities (PDF, Word)
- [ ] Custom embedding models

---

⭐ **Star this repository if you find it helpful!**

🔗 **Share it with others who might benefit from this RAG solution!**

📹 **Watch the demo video**: https://youtu.be/FRPM9zABVVg
