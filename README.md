# LLM Papers RAG App Streamlit

Streamlit web application for querying AI research papers using RAG technology. Interactive interface to ask questions about transformer, GPT-4, Gemini, and other machine learning papers with LlamaIndex.

---

## 🚀 Features

- **Multiple Node Parsing Strategies**: Sentence splitter, hierarchical, and sentence window parsers
- **Embedding Model Comparison**: OpenAI, Gemini, Cohere, and HuggingFace embeddings
- **Advanced Retrieval Techniques**: Auto-retriever, auto-merging retriever, and sub-question query engine
- **Performance Optimization**: Systematic evaluation of different RAG configurations
- **Interactive Web Interface**: Streamlit-based query interface with ngrok tunneling
- **Research Paper Analysis**: Specialized for AI/ML research papers (Transformer, GPT-4, Gemini, etc.)

---

## 📂 Project Structure

```
advanced-rag-system/
├── Capston_Project.ipynb          # Main notebook with complete RAG implementation
├── app.py                         # Streamlit web application
├── data/                          # Research papers directory
│   ├── attention_paper.pdf
│   ├── gemini_paper.pdf
│   ├── gpt4.pdf
│   ├── instructgpt.pdf
│   └── mistral_paper.pdf
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/advanced-rag-system.git
cd advanced-rag-system
```

2. Install required dependencies:
```bash
pip install llama-index
pip install transformers
pip install llama-index-embeddings-huggingface
pip install llama-index-embeddings-gemini
pip install llama-index-llms-cohere
pip install llama-index-embeddings-cohere
pip install streamlit==1.32.2
pip install pyngrok==7.1.5
pip install openai==1.55.3
pip install httpx==0.27.2
```

3. Set up API keys:
```python
# Add your API keys to Google Colab secrets or environment variables
GOOGLE_API_KEY = "your_gemini_key"
HUGGINGFACE_API_KEY = "your_huggingface_key" 
COHERE_API_KEY = "your_cohere_key"
OPENAI_API_KEY = "your_openai_key"
NGROK_AUTH_TOKEN = "your_ngrok_token"
```

---

## 🔧 Usage

### Running the Jupyter Notebook

1. Open `Capston_Project.ipynb` in Google Colab or Jupyter
2. Mount Google Drive and ensure your data folder is accessible
3. Run cells sequentially to:
   - Load and parse documents
   - Compare different node parsing strategies
   - Evaluate embedding models
   - Test various retrieval techniques
   - Analyze performance results

### Running the Streamlit App

1. Execute the Streamlit app cells in the notebook:
```python
!streamlit run app.py --server.port=8989 &>./logs.txt &
```

2. Set up ngrok tunnel for external access:
```python
from pyngrok import ngrok
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
ngrok_tunnel = ngrok.connect(8989)
print("Streamlit App:", ngrok_tunnel.public_url)
```

3. Access the web interface through the provided ngrok URL

---

## 📊 Results

### Key Findings:

- **Best Node Parser**: SentenceSplitter with chunk_size=500 and chunk_overlap=50
- **Best Embedding Model**: OpenAI's text-embedding-3-small
- **Best Retrieval Method**: Sub-Question Query Engine for complex multi-document queries
- **Performance**: Successfully handles 95 research documents with 205 optimized nodes

### Query Examples:

- ✅ "What is Scaled Dot-Product Attention?" 
- ✅ "What is Multi-Head Attention?"
- ✅ "What are 3 types of regularization used during training?"
- ✅ "Which optimizer is mentioned in the paper?"

---

## 📋 Requirements

- **Python**: 3.9+
- **GPU**: Recommended (T4 or better for faster processing)
- **Memory**: 8GB+ RAM recommended
- **Libraries**:
  - llama-index
  - transformers  
  - openai
  - streamlit
  - pyngrok
  - tiktoken
  - pathlib

---

## 🧪 Experimental Design

The project includes systematic evaluation of:

1. **Node Parsing Strategies**:
   - SimpleFileNodeParser
   - SentenceSplitter (various chunk sizes)
   - SentenceWindowNodeParser
   - HierarchicalNodeParser

2. **Embedding Models**:
   - OpenAI (text-embedding-3-small, text-embedding-ada-002)
   - Google Gemini (text-embedding-004)
   - Cohere (embed-english-v3.0)
   - HuggingFace (BAAI/bge-small-en-v1.5, WhereIsAI/UAE-Large-V1)

3. **Retrieval Techniques**:
   - Standard Vector Store Retrieval
   - Auto-Retriever with metadata filtering
   - Auto-Merging Retriever
   - Sub-Question Query Engine

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 🙏 Acknowledgments

- Research papers used for testing from leading AI institutions
- LlamaIndex framework for RAG implementation
- OpenAI, Google, Cohere, and HuggingFace for embedding models
- Streamlit for web interface development
