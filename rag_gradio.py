import gradio as gr
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from collections import Counter
import os

# -------------------------------
# 1. Load IMDb dataset
# -------------------------------
print("Loading IMDb dataset from Hugging Face...")
dataset = load_dataset("imdb")
texts = dataset["train"]["text"][:5000]  # subset for demo
print(f"Loaded {len(texts)} reviews.")

# -------------------------------
# 2. Load / Create SentenceTransformer embeddings
# -------------------------------
MODEL_DIR = "models/all-MiniLM-L6-v2"
if not os.path.exists(MODEL_DIR):
    print("Downloading SentenceTransformer model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embedder.save(MODEL_DIR)
else:
    print("Loading SentenceTransformer model from disk...")
    embedder = SentenceTransformer(MODEL_DIR)

print("Creating embeddings...")
text_embeddings = embedder.encode(texts, convert_to_numpy=True)

# -------------------------------
# 3. FAISS index
# -------------------------------
dimension = text_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(text_embeddings)
print("FAISS index ready.")

# -------------------------------
# 4. Load sentiment-analysis pipeline
# -------------------------------
MODEL_SENTIMENT_DIR = "models/sentiment_model"
if not os.path.exists(MODEL_SENTIMENT_DIR):
    print("Downloading sentiment model...")
    sentiment_model = pipeline("sentiment-analysis", truncation=True, max_length=512)
    sentiment_model.model.save_pretrained(MODEL_SENTIMENT_DIR)
    sentiment_model.tokenizer.save_pretrained(MODEL_SENTIMENT_DIR)
else:
    print("Loading sentiment model from disk...")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SENTIMENT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SENTIMENT_DIR)
    sentiment_model = TextClassificationPipeline(
        model=model, tokenizer=tokenizer, truncation=True, max_length=512
    )

# -------------------------------
# 5. RAG + Sentiment function
# -------------------------------
MAX_CHARS = 1000

def truncate_text(text, max_chars=MAX_CHARS):
    return text[:max_chars]

def rag_sentiment(query, k=5):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k)
    retrieved_texts = [truncate_text(texts[i]) for i in I[0]]

    sentiments = sentiment_model(retrieved_texts)
    results = []
    labels = []
    for text, sentiment in zip(retrieved_texts, sentiments):
        results.append({
            "text": text,
            "label": sentiment["label"],
            "score": round(sentiment["score"], 3)
        })
        labels.append(sentiment["label"])

    counts = Counter(labels)
    overall_label, count = counts.most_common(1)[0]
    overall_sentiment = f"Overall Sentiment: {overall_label} ({count}/{k} reviews)"
    return results, overall_sentiment

# -------------------------------
# 6. Gradio UI
# -------------------------------
def gradio_ui(query):
    results, overall_sentiment = rag_sentiment(query, k=5)
    display = ""
    for r in results:
        display += f"Text: {r['text']}\nSentiment: {r['label']} (confidence: {r['score']})\n\n"
    display += overall_sentiment
    return display

with gr.Blocks() as demo:
    gr.Markdown("## 🎯 Sentiment Analysis with RAG + FAISS + Hugging Face (IMDb Dataset)")
    gr.Markdown("IMDb dataset URL: [https://huggingface.co/datasets/imdb](https://huggingface.co/datasets/imdb)")

    query_input = gr.Textbox(placeholder="Enter your text here", label="Your Text")
    output_box = gr.Textbox(label="Retrieved Reviews + Sentiment", lines=20)
    query_input.submit(gradio_ui, query_input, output_box)

# -------------------------------
# 7. Launch
# -------------------------------
demo.launch(server_name="0.0.0.0", server_port=7860)
