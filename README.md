# Systematic Review Agent Pipeline

A **multi-agent Systematic Review pipeline** built with **DSPy**, **Typer**, and **Rich**.  
It performs the following steps:  

1. **Problem Formulation Agent** – Refines a research question into structured keywords.  
2. **Data Retrieval Agent** – Fetches papers from Crossref.  
3. **Data Prescreen Agent** – Filters retrieved papers based on inclusion criteria.  
4. **PDF Retrieval Agent** – Simulates retrieval of PDF links for filtered papers.  
5. **Data Extraction & Synthesis Agent** – Summarizes key insights using Gemini LLM.  

The output is a **rich CLI experience** with colored tables and panels.  

---

## 🚀 Features

- Modular DSPy agents sharing data sequentially.  
- Rich CLI output for easy visualization.  
- Gemini-powered reasoning for problem formulation and synthesis.  
- Fully local setup — no external web UI needed.  

---

## 📦 Requirements

- Python 3.10+  
- DSPy (latest version)  
- Typer  
- Rich  
- Requests  
- `python-dotenv` (for API keys)  

---

## 🔧 Installation

1. **Clone the repository**

```
git clone https://github.com/yourusername/systematic-review-agent.git
cd systematic-review-agent
```

2. **Create and activate a virtual environment**

```
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. **Install Dependencies**

```
pip install -r requirements.txt
```

4. **Add your API keys**

Create a .env file in the project root:

```
GOOGLE_API_KEY=your_google_or_gemini_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

5. **⚡Running the Pipeline**

```
python main.py "What are the recent advances in gene therapy for hemophilia?"
```
