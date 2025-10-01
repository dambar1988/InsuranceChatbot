# backend.py
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------
# Load embedding and generation models
# -----------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"  # faster on CPU

print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Loading generation model...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
generator = pipeline(
    "text2text-generation", model=gen_model, tokenizer=tokenizer, device=-1
)

# -----------------------
# Load documents
# -----------------------
with open("documents.json", "r", encoding="utf-8") as f:
    DOCUMENTS = json.load(f)

# -----------------------
# Create FAISS index
# -----------------------
doc_embeddings = np.array([embedder.encode(doc["content"]) for doc in DOCUMENTS]).astype("float32")
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(doc_embeddings)
TOP_K = 2

# -----------------------
# Retrieve relevant documents from FAISS
# -----------------------
def retrieve(question, top_k=TOP_K):
    qvec = np.array([embedder.encode(question)]).astype("float32")
    distances, indices = index.search(qvec, top_k)
    results = [DOCUMENTS[i] for i in indices[0]]
    return results

# -----------------------
# Generate answer using RAG
# -----------------------
BASE_PROMPT = """You are Allianz Australia's internal insurance chatbot.
Answer the question using only the context below. Do not make up answers.
If not available, reply: "Not available in Allianz knowledge base."

CONTEXT:
{context}

QUESTION: {question}
"""

def generate_answer(question):
    retrieved = retrieve(question)
    context = "\n\n".join([f"{d['title']}: {d['content']}" for d in retrieved])
    prompt = BASE_PROMPT.format(context=context, question=question)
    response = generator(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
    return response, retrieved
