# InsuranceChatbot
Insurance Chatbot with RAG


# InsuranceChatbot

**InsuranceChatbot** is an internal chatbot designed for **Any Insurance compony employees**. It provides instant, accurate answers to insurance-related questions using Allianz’s internal knowledge base. The chatbot combines **RAG (Retrieval-Augmented Generation)** architecture with **FAISS** for document retrieval and **Hugging Face Transformers** for text generation.

---

## Project Overview

This project solves the problem of **employee access to insurance information** by automating responses to policy, claim, and coverage queries.  

**Key points:**

1. Employees can ask questions via a **web-based ChatGPT-style interface**.
2. The chatbot retrieves **relevant insurance documents** from the knowledge base using embeddings.
3. A **generation model** (FLAN-T5) formulates natural, context-aware responses.
4. Ensures employees only get **verified Allianz information**; it never hallucinates answers.

---

## Features

- Retrieve Allianz insurance documents using **FAISS embeddings**.
- Generate accurate answers with **Hugging Face text2text models**.
- Easy-to-use **web UI** for chat.
- Fully local deployment: no cloud dependency needed.
- Scalable: can add more documents to `documents.json` anytime.

---

## Project Architecture

User
└─> Web UI (Flask - app.py)
└─> Backend (backend.py)
├─ Embedding Model: Sentence-Transformers (all-MiniLM-L6-v2)
├─ FAISS Index: Stores document vectors for similarity search
├─ Generation Model: Hugging Face FLAN-T5
└─ documents.json: Internal knowledge base


**How it works:**

1. **User Question:** Employee types a question in the web UI.
2. **Embedding & Retrieval:** `backend.py` converts the question into an embedding and searches the FAISS index for the most relevant documents.
3. **RAG Generation:** The retrieved document content is sent to the generation model (FLAN-T5) which formulates a coherent answer.
4. **Response:** Answer is displayed in the web interface.

---

## Project Structure

InsuranceChatbot/
├── backend.py # Handles embeddings, FAISS retrieval, and response generation
├── app.py # Flask app with ChatGPT-style UI
├── documents.json # Insurance knowledge base (editable)
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files (venv, cache, etc.)
└── README.md # Project explanation (this file)


---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/dambar1988/InsuranceChatbot.git
cd InsuranceChatbot

2. Create and activate a virtual environment:

# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python -m venv .venv
source .venv/bin/activate

3. Install dependencies:

pip install -r requirements.txt
//add all jar inside

::::Usage::::::::

1. Run the Flask app:
  python app.py

2. Open your browser:
  http://localhost:5100/

3. Type insurance-related questions.
  The chatbot retrieves relevant documents from documents.json and generates accurate responses.

::::::::::::Dependencies::::::::::
Python 3.10+
Flask – Web interface
FAISS (faiss-cpu) – Vector similarity search
Sentence Transformers – Document embeddings
Transformers (Hugging Face) – Text generation
Torch – Backend for transformers
NumPy – Vector operations


:::::::::::::::::How to Contribute:::::::::::::

Fork the repository.
Make changes in a new branch.
Add new documents or improve backend logic.
Commit your changes:

git add .
git commit -m "Describe your change"
git push origin your-branch










