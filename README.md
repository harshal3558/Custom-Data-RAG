# 🧠 Custom RAG: Intelligent PDF Retrieval System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-black?logo=flask&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C?logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama3--70B-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A state-of-the-art **Retrieval-Augmented Generation (RAG)** application designed to transform static PDF documents into interactive knowledge bases. By combining the speed of **Groq Cloud (Llama-3)** with the modularity of **LangChain**, this system provides accurate, context-aware answers in milliseconds.

---

## 📋 Table of Contents
- [📖 About the Project](#-about-the-project)
- [✨ Key Features](#-key-features)
- [🛠️ Tech Stack](#-tech-stack)
- [🏗️ Project Architecture](#-project-architecture)
- [📈 Monitoring & Tracking](#-monitoring--tracking)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Installation](#local-installation)
- [🔌 API Endpoints](#-api-endpoints)
- [🤝 Contributing](#-contributing)
- [📩 Contact](#-contact)

---

## 📖 About the Project

The **Custom RAG System** is built to bridge the gap between static enterprise data and user queries. It features a robust pipeline that automates document ingestion, semantic chunking, and vector indexing. 

Whether you're processing technical manuals (like `DogTraining101.pdf` included in this repo) or dense research papers, the system retrieves only the most relevant sections to ground its AI responses, completely eliminating hallucinations.

---

## ✨ Key Features
- **Instant PDF Ingestion**: Upload documents via a web portal for automatic background indexing.
- **Lightning-Fast Generation**: Powered by **Groq**, delivering Llama-3-70B responses at unprecedented speeds.
- **Persistent Vector Storage**: Uses **ChromaDB** to ensure your document embeddings are saved across sessions.
- **Hybrid Search Capabilities**: Fine-tuned retrieval logic with **Sentence-Transformers** for high semantic accuracy.
- **Experiment Monitoring**: Fully integrated with **MLflow** to track ingestion metrics and LLM performance.
- **Clean UI**: A responsive Flask-driven frontend for seamless document interaction.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Core AI** | LangChain, Llama-3 (via Groq API) |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Backend** | Python, Flask |
| **Vector DB** | ChromaDB (Persistent) |
| **Document Processing** | PyMuPDF (Fitz), PyPDF |
| **Observability** | MLflow, DagsHub |
| **Logging & Stats** | NumPy, Custom Exception & Logger |

---

## 🏗️ Project Architecture

```text
Custom_RAG/
├── app.py                  # Flask Web Application entry point
├── src/                    # Source Directory
│   ├── components/         # Modular pipeline components
│   │   ├── data_ingestion.py    # PDF loading and chunking
│   │   ├── data_transformation.py # Vector embedding and indexing
│   │   └── model_trainer.py
│   ├── pipeline/           # High-level pipeline execution
│   │   ├── prediction_pipeline.py # RAG query logic (Retrieval + LLM)
│   │   └── training_pipeline.py
│   ├── logger.py           # Custom logging utility
│   └── exception.py        # Custom error handling
├── data/                   # Data Storage
│   ├── uploads/            # Uploaded PDF files
│   └── vector_store/       # Persistent ChromaDB collection
├── templates/              # HTML frontend files
├── static/                 # CSS/JS assets
├── requirements.txt        # Backend dependencies
└── setup.py                # Local package installation
```

---

## 📈 Monitoring & Tracking

This project implements **MLflow** for rigorous tracking of both data processing and inference quality.

**Components Tracked:**
- **Indexing Pipeline**: Logs `chunk_size`, `model_name`, and `num_chunks` to evaluate indexing efficiency.
- **Inference Monitoring**: Captures `query`, `answer_length`, and `num_retrieved_docs` to monitor performance and context usage.

To view the dashboard after running queries:
```bash
mlflow ui
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A [Groq API Key](https://console.groq.com/keys)
- (Optional) DagsHub account for remote artifact tracking

### Local Installation

1. **Clone the project**
   ```bash
   git clone https://github.com/harshal3558/Custom-Data-RAG.git
   cd Custom_RAG
   ```

2. **Create a Environment**
   ```bash
   # Windows
   python -m venv cragenv
   .\cragenv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file in the root:
   ```env
   GROQ_API_KEY=your_key_here
   GROQ_MODEL_NAME=llama-3.3-70b-versatile
   ```

5. **Run the App**
   ```bash
   python app.py
   ```
   Open `http://localhost:5000` in your browser.

---

## 🔌 API Endpoints

- **`GET /`**: Renders the application homepage.
- **`POST /upload`**: Takes a PDF file, executes chunking and vector store ingestion.
- **`POST /query`**: Accepts a JSON query and returns an AI-generated answer based on retrieved documents.

**Input Sample (Query):**
```json
{
  "query": "What are the common dog behaviors discussed?"
}
```

---

## 🤝 Contributing

We value contributions!
1. **Fork** the repository.
2. **Create** your feature branch (`git checkout -b feature/NewComponent`).
3. **Commit** your changes (`git commit -m 'Add NewComponent'`).
4. **Push** to the branch (`git push origin feature/NewComponent`).
5. **Open** a Pull Request.

---

## 📩 Contact

**Harshal** - [harshal3558@gmail.com](mailto:harshal3558@gmail.com)

Project Link: [https://github.com/harshal3558/Custom-Data-RAG](https://github.com/harshal3558/Custom-Data-RAG)