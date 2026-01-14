# retriever.py

import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import tiktoken

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
TOP_K = 10

INDEX_FILE = "vector.index"
META_FILE = "metadata.json"

# Local LLM config (as given)
LLM_URL = "http://aigateway.iagami.com:4000/v1/chat/completions"
LLM_MODEL = "oci/google.gemini-2.5-flash-lite"
# LLM_MODEL = "oci/google.gemini-2.5-flash"
# LLM_MODEL = "oci/xai.grok-4"
LLM_API_KEY = "sk-aphZ6iCssH6F97y_wj_3Hw"
LLM_API_KEY1 = "sk-iaga2024mi2025tech"

model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# LOAD VECTOR STORE
# -----------------------------
def load_store():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def is_overview_query(query: str) -> bool:
    overview_keywords = [
        "explain",
        "overview",
        "summary",
        "what is",
        "tell me about",
        "describe"
    ]
    q = query.lower()
    return any(k in q for k in overview_keywords)

# -----------------------------
# RETRIEVAL
# -----------------------------
def retrieve(query, index, metadata):
    if is_overview_query(query):
        # Find the most relevant document first
        q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(q_vec, 1)
        top_chunk = metadata[indices[0][0]]
        doc_name = top_chunk["document_title"]

        # Fetch ALL chunks from the same document
        results = [
            m for m in metadata
            if m["document_title"] == doc_name
        ]

        print(f"📄 Overview query detected. Returning {len(results)} chunks from document: {doc_name}")
        return results

    # else → normal semantic retrieval
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_vec, TOP_K)

    return [metadata[i] for i in indices[0] if i != -1]


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
        ],
        "temperature": 2
         
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


def count_tokens(text: str) -> int:
    """
    Approximate token count using cl100k_base.
    Works safely for Gemini, Claude, OpenAI models.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
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
You are an internal HR knowledge assistant.

Your responsibilities:
1. Understand the user’s intent from the question.
2. Use the provided context as the ONLY source of knowledge.
3. Reason over the HR policies to produce a clear, policy-aligned answer.

HOW TO INTERPRET HR POLICIES:
- HR policies define entitlements, eligibility, limits, and calculation rules.
- If a policy explicitly mentions proration, monthly entitlement, or proportional allocation,
  you MAY logically calculate or explain outcomes based on the policy text.
- Logical conclusions are allowed IF they are directly supported by policy statements
  (for example: “1 day per completed month”, “prorated for the calendar year”).
- Clearly explain the reasoning behind such conclusions.
- If the policy defines a calendar period (for example, “Jan–Dec”),
  you MAY count completed months within that period to explain
  prorated or monthly-based entitlements.
- You may assume standard calendar months only when the policy
  explicitly defines the leave year.
- If the policy provides a clear calculation basis (for example:
  “1 day per completed month” or “prorated for the calendar year”),
  you MUST explain how the calculation works using a simple example
  (without inventing new rules).
- HR policies may define a rule across multiple clauses or sections.
  If a calculation or entitlement can be derived only by combining
  two or more policy statements, you MUST synthesize them and
  provide the resulting explanation.


WHAT NOT TO DO:
- Do NOT use external laws, assumptions, or common HR practices not stated in the policy.
- Do NOT invent numbers or rules that are not supported by the policy text.
- Do NOT ignore relevant sections that allow proportional or calculated outcomes.

HANDLING PARTIAL OR MISSING INFORMATION:
- If the policy explicitly defines a rule → apply it logically.
- If the policy is silent or ambiguous → clearly state the limitation.
- Absence of a policy should be treated as “not provided”.



ONLY respond with:
"I could not find this information in the provided documents."
IF AND ONLY IF:
- The policy provides no direct rule,
- AND no reasonable, policy-supported conclusion can be drawn.

Context:
{context}

Question:
{query}

Answer:
""".strip()
    print(f"🧠 Prompt to LLM: {prompt}")
    print(f"🧠 Prompt tokens: {count_tokens(prompt)}")
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
