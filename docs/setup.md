# Setup Guide

## Prerequisites

- Python 3.10+
- Virtual environment (recommended)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd Adaptive-RAG-With-LangChain
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create `.env` file:
   ```
   MISTRAL_API_KEY=your_key
   TAVILY_API_KEY=your_key
   COHERE_API_KEY=your_key
   PINECONE_API_KEY=your_key
   PINECONE_INDEX_NAME=adaptive-rag
   ```

5. **Place PDFs**
   ```bash
   mkdir -p data/pdfs
   # Add your PDF files to data/pdfs/
   ```

## Running

### Demo
```bash
python scripts/demo.py
```

### Web API
```bash
python run_app.py
```

### Initialize Pinecone Index
```bash
python scripts/setup_index.py
```
