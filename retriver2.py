import json
import os
import numpy as np
import requests
import os
from sentence_transformers import CrossEncoder, SentenceTransformer
import tiktoken
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sheets_logger import log_answer


from dotenv import load_dotenv
load_dotenv()

import time
from functools import wraps

# -----------------------------
# CONFIG
# -----------------------------
CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "hr_policies"

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
TOP_K = 20
TOP_N_RERANK = 10


LLM_URL = os.getenv("LLM_URL1")
LLM_MODEL = os.getenv("LLM_MODEL1")
LLM_API_KEY = os.getenv("LLM_API_KEY1")

model = SentenceTransformer(MODEL_NAME)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def timer_decorator(func):
    """Decorator to measure and print the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        
        # Check if result is a generator
        if hasattr(result, '__iter__') and hasattr(result, '__next__'):
            # Wrap the generator to measure execution time
            def timed_generator():
                try:
                    for item in result:
                        yield item
                finally:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"\n {func.__name__} execution time: {elapsed_time:.4f} seconds\n")
            return timed_generator()
        else:
            # Non-generator: measure time immediately
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\n  {func.__name__} execution time: {elapsed_time:.4f} seconds\n")
            return result
    return wrapper

# -----------------------------
# LOAD VECTOR STORE
# -----------------------------


embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True}
)


vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}  
)


# -----------------------------
# RETRIEVAL
# -----------------------------
@timer_decorator

def retrieve(query: str):
    #  OVERVIEW QUERY (STRUCTURAL)
        
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=TOP_K)

    # print("\n========= RETRIEVED CHUNKS (NO FILTERING) =========\n")

    chunks = []
    #     print(f"CHUNK #{idx}")
    #     print(f"Vector Score : {score:.4f} (distance)")
    #     print(f"Document     : {d.metadata.get('document_title')}")
    #     print(f"Section      : {d.metadata.get('section_title')}")
    #     print(f"Page         : {d.metadata.get('page_number')}")
    #     print("-" * 70)
    #     print(d.page_content)
    #     print("-" * 70 + "\n")
    for idx, (d, score) in enumerate(docs_with_scores, start=1):
        chunks.append({
            "document_title": d.metadata["document_title"],
            "section_title": d.metadata["section_title"],
            "page_number": d.metadata["page_number"],
            "content": d.page_content,
            "vector_score": score
        })

    # print("===============================\n")
    return chunks

  

# -----------------------------
# RERANK CHUNKS (RANK-BASED ONLY)
# -----------------------------
@timer_decorator
def rerank_chunks(query: str, chunks: list, top_n: int):
    if not chunks:
        return chunks

    pairs = [(query, c["content"]) for c in chunks]
    scores = reranker.predict(pairs)

    for c, score in zip(chunks, scores):
        c["rerank_score"] = float(score)

    chunks = sorted(
        chunks,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return chunks[:top_n]

# -----------------------------
# LLM CALL (LOCAL)
# -----------------------------

@timer_decorator
def call_llm_stream(prompt: str):
    """
    Streams output from OpenAI-compatible /v1/chat/completions endpoint.
    Prints token chunks as they arrive.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": True,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }

    with requests.post(
        LLM_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            # Expected format: "data: {...}"
            if line.startswith("data: "):
                data_str = line[len("data: ") :].strip()

                # Stream end signal
                if data_str == "[DONE]":
                    break

                try:
                    data_json = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # OpenAI-style streaming delta
                delta = (
                    data_json.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )

                if delta:
                    yield delta


# -----------------------------
# RAG ANSWERING (STREAM)
# -----------------------------
@timer_decorator
def answer_question_stream(query):
    chunks = retrieve(query)
    if not chunks:
        yield "I could not find this information in the provided documents."
        return
    chunks = rerank_chunks(query, chunks, top_n=TOP_N_RERANK)
    print(f" Using {len(chunks)} chunks for context.")
    context = "\n\n".join(
        f"[Document: {c['document_title']} | "
        f"Section: {c['section_title']} | "
        f"Page: {c['page_number']}]\n{c['content']}"
        for c in chunks
    )

    prompt = f"""
You are an internal HR knowledge assistant.
IMPORTANT: Use simple, employee-friendly language. Avoid formal HR/legal terms unless necessary.

Your responsibilities:
1. Understand the user’s intent.
2. Use the provided context as the ONLY source of truth for company policy rules.
3. Answer in a clear, policy-aligned way.
4. If the user asks a general HR term (example: "financial year", "sandwich leave"), you may give a short standard definition even if not in the documents — but clearly label it as a general definition.

HOW TO INTERPRET HR POLICIES:
- HR policies define entitlements, eligibility, limits, and calculation rules.
- If the policy explicitly mentions proration / monthly entitlement / proportional allocation / calculation rules,
  you MAY calculate outcomes only when the rule is clearly stated.
- Do NOT invent formulas or numbers not present in the context.
- If the user asks “remaining leave” and entitlement + usage are clearly provided, you MAY compute:
  remaining = entitlement - used.

FOLLOW-UP QUESTIONS (IMPORTANT):
- If the user asks a short follow-up (example: “how many days?”, “how much?”, “remaining?”),
  treat it as a follow-up to the most recent topic in the conversation.
- Do NOT list all leave types unless the user explicitly asks for a full list or comparison.
- If the leave type/topic is unclear, ask one short clarification question.

TERMINOLOGY MAPPING (IMPORTANT):
- Users may use informal terms (example: “sandwich leave”, “LOP cut”, “salary deduction”).
- If the exact term is not present, but the policy describes the same concept,
  answer using the closest matching rule and mention the term is not explicitly used.

ANSWER STYLE (VERY IMPORTANT):
- Be straight-forward and minimal by default.
- Give the direct answer first.
- ONLY explain “why/how” if the user asks for explanation.
- Do NOT add long lists unless requested.

MISSING POLICY HANDLING (IMPORTANT):
- If the user asks about a company-specific policy (example: paternity leave, notice period, bonus rule)
  and it is NOT mentioned in the provided context:
  - Do NOT say only "I could not find this information..."
  - Instead reply like this:
    "This is not mentioned in the provided policy documents, so it is not confirmed. Please check with HR."
  - Keep it short.

REFERENCE / SOURCE CITATION (IMPORTANT):
- If your answer uses the provided policy context, add a reference at the end in this format:
  Reference: [Document: ... | Section: ... | Page: ...]
- Use only 1–2 most relevant references.
- If you give a general HR definition (not from policy docs), write:
  Reference: General HR definition (not from company policy documents)

OUTPUT FORMAT:
- Start with a 1-line direct answer.
- Add 2–3 bullet points ONLY if needed.
- Add Reference at the end when applicable.

Context:
{context}

Question:
{query}

Answer:
""".strip()

    print(f"\n Prompt tokens: {count_tokens(prompt)}")
    # print(f"prompt: {prompt}")
    print("\n Answer (streaming):\n")

    # Stream tokens
    for chunk in call_llm_stream(prompt):
        yield chunk


def count_tokens(text: str) -> int:
    """
    Approximate token count using cl100k_base.
    Works safely for Gemini, Claude, OpenAI models.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

if __name__ == "__main__":
    print(" RAG Retriever ready (type 'exit' to quit)")

    while True:
        query = input("\nAsk a question: ").strip()
        if query.lower() in ("exit", "q"):
            break

        # 🔹 LLM STREAMING
        try:
            full_answer = ""

            for token in answer_question_stream(query):
                print(token, end="", flush=True)
                full_answer += token

            print("\n")

        except Exception as e:
            print("\n Error during LLM streaming")
            print(e)
            continue  

        try:
            print("DEBUG: about to save to Google Sheets")
            log_answer(query, full_answer)
            print(" Answer saved to Google Sheets")

        except Exception as e:
            print("\nError during Google Sheets logging")