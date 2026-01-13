# retriever.py

import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
TOP_K = 5

INDEX_FILE = "vector.index"
META_FILE = "metadata.json"

# Local LLM config (as given)
LLM_URL = "http://aigateway.iagami.com:4000/v1/chat/completions"
# LLM_MODEL = "oci/google.gemini-2.5-flash-lite"
LLM_MODEL = "oci/google.gemini-2.5-flash"
LLM_API_KEY = "sk-aphZ6iCssH6F97y_wj_3Hw"

model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# LOAD VECTOR STORE
# -----------------------------
def load_store():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# -----------------------------
# RETRIEVAL
# -----------------------------
def retrieve(query, index, metadata):
    q_vec = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(q_vec, TOP_K)

    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(metadata[idx])
    print(f"🔍 Retrieved {len(results)} chunks for the query.")
    print(f"   ↳ Top chunk from document: {results[0]['document_title']}" if results else "   ↳ No chunks found.")
    print(results)
    return results

# -----------------------------
# LLM CALL (LOCAL)
# -----------------------------
def call_llm(prompt: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }

    response = requests.post(
        LLM_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]

# -----------------------------
# RAG ANSWERING
# -----------------------------
def answer_question(query, index, metadata):
    chunks = retrieve(query, index, metadata)

    if not chunks:
        return "I could not find this information in the provided documents."

    context = "\n\n".join(
        f"[Document: {c['document_title']} | "
        f"Section: {c['section_title']} | "
        f"Page: {c['page_number']}]\n{c['content']}"
        for c in chunks
    )

    prompt = f"""
You are an internal knowledge assistant.

Answer ONLY using the information in the context below.
You may combine or summarize multiple statements from the context
if the answer requires aggregation.
Do NOT use outside knowledge.
If the information is still insufficient, say:
"I could not find this information in the provided documents."

Context:
{context}

Question:
{query}

Answer:
""".strip()

    return call_llm(prompt)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    index, metadata = load_store()

    print("🤖 RAG Retriever ready (type 'exit' to quit)")

    while True:
        query = input("\nAsk a question: ").strip()
        if query.lower() == "exit" or query.lower() == "q":
            break

        try:
            answer = answer_question(query, index, metadata)
            print("\n✅ Answer:\n")
            print(answer)
        except Exception as e:
            print("\n❌ Error during LLM call:")
            print(str(e))
