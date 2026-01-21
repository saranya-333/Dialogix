# retriever.py

import json

import numpy as np
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer
import tiktoken
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


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


# Local LLM config (as given)
LLM_URL = "http://aigateway.iagami.com:4000/v1/chat/completions"
# LLM_MODEL = "oci/google.gemini-2.5-flash-lite"
LLM_MODEL = "oci/google.gemini-2.5-flash"
# LLM_MODEL = "oci/xai.grok-4"
LLM_API_KEY = "sk-aphZ6iCssH6F97y_wj_3Hw"
LLM_API_KEY1 = "sk-iaga2024mi2025tech"

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
                    print(f"\n⏱️  {func.__name__} execution time: {elapsed_time:.4f} seconds\n")
            return timed_generator()
        else:
            # Non-generator: measure time immediately
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\n⏱️  {func.__name__} execution time: {elapsed_time:.4f} seconds\n")
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
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
@timer_decorator
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
@timer_decorator

def retrieve(query: str):
    #  OVERVIEW QUERY (STRUCTURAL)
    if is_overview_query(query):
        print("\n OVERVIEW QUERY DETECTED\n")

        docs_with_scores = vectorstore.similarity_search_with_score(query, k=1)
        if not docs_with_scores:
            return []

        doc, score = docs_with_scores[0]
        print(f"Top document score (distance): {score:.4f}\n")

        document_title = doc.metadata.get("document_title")

        all_docs = vectorstore.get(
            where={"document_title": document_title}
        )

        chunks = []
        for content, meta in zip(all_docs["documents"], all_docs["metadatas"]):
            chunks.append({
                "document_title": meta["document_title"],
                "section_title": meta["section_title"],
                "page_number": meta["page_number"],
                "content": content
            })

        print(f" Returning {len(chunks)} chunks from document: {document_title}\n")
        return chunks

    # 🔹 SPECIFIC QUERY
    
    
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=TOP_K)

    # print("\n========= RETRIEVED CHUNKS (NO FILTERING) =========\n")

    chunks = []
    for idx, (d, score) in enumerate(docs_with_scores, start=1):
    #     print(f"CHUNK #{idx}")
    #     print(f"Vector Score : {score:.4f} (distance)")
    #     print(f"Document     : {d.metadata.get('document_title')}")
    #     print(f"Section      : {d.metadata.get('section_title')}")
    #     print(f"Page         : {d.metadata.get('page_number')}")
    #     print("-" * 70)
    #     print(d.page_content)
    #     print("-" * 70 + "\n")

      chunks.append({
            "document_title": d.metadata["document_title"],
            "section_title": d.metadata["section_title"],
            "page_number": d.metadata["page_number"],
            "content": d.page_content,
            "vector_score": score
        })

    # print("===============================\n")
    return chunks

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

@timer_decorator
def call_llm_stream(prompt: str):
    """
    Streams output from OpenAI-compatible /v1/chat/completions endpoint.
    Prints token chunks as they arrive.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 2,
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

        # Each line is typically: b"data: {...}\n"
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
    print(f"🔍 Using {len(chunks)} chunks for context.")
    context = "\n\n".join(
        f"[Document: {c['document_title']} | "
        f"Section: {c['section_title']} | "
        f"Page: {c['page_number']}]\n{c['content']}"
        for c in chunks
    )

    prompt = f"""
You are an internal HR knowledge assistant.
IMPORTANT:Use simple, employee-friendly language instead of formal or technical HR terminology.

Your responsibilities:
1. Understand the user’s intent from the question.
2. Use the provided context as the ONLY source of knowledge for company policy rules.
3. Reason over the HR policies to produce a clear, policy-aligned answer.
4. For general HR terminology questions, provide a short, standard definition even if not present in the documents.

HOW TO INTERPRET HR POLICIES:
- HR policies define entitlements, eligibility, limits, and calculation rules.
- If a policy explicitly mentions proration, monthly entitlement, proportional allocation, or calculation rules,
  you MAY logically calculate or explain outcomes based on the policy text.
- Logical conclusions are allowed ONLY IF they are directly supported by policy statements
  (for example: “1 day per completed month”, “prorated for the calendar year”).
- If the policy provides a clear calculation basis, you MUST apply it correctly.
- If the user asks “remaining leave”, and the entitlement is provided in policy,
  you MAY calculate: remaining = entitlement - used (only when both are clearly available).
- If the policy defines a calendar period (for example, “Jan–Dec”),
  you MAY count completed months within that period to explain prorated or monthly-based entitlements.
- If a calculation or entitlement requires combining multiple clauses/sections,
  you MUST synthesize them into one clear explanation.

FOLLOW-UP QUESTIONS (IMPORTANT):
- If the user asks a short/unclear follow-up question (example: “how many days?”, “how much?”, “remaining?”),
  treat it as a follow-up to the most recent leave type/topic discussed in the conversation.
- Do NOT list all leave types unless the user explicitly asks for a full list or comparison.
- If the leave type/topic is not clear, ask a short clarification question.

IMPORTANT OVERRIDE FOR TERMINOLOGY QUESTIONS:
- Users may ask using informal HR terms (example: “sandwich leave”, “LOP cut”, “salary deduction”, “salary hike”, etc).
- If the exact term is not present in the policy, but the policy contains a rule that matches the concept,
  you MUST answer using the closest matching policy rule.
- In such cases, do NOT use the fallback message.

ANSWER STYLE (VERY IMPORTANT):
- Be straight-forward and minimal by default.
- Give the direct answer first.
- ONLY explain the reasoning if the user explicitly asks “why”, “how”, “explain”, or “details”.
- Do NOT add extra information, examples, or long lists unless the user asks.

WHAT NOT TO DO:
- Do NOT use external laws, assumptions, or common HR practices not stated in the policy.
- Do NOT invent numbers, percentages, salary formulas, or payroll divisors unless explicitly stated in the policy text.
- Do NOT ignore relevant sections that affect the answer.

HANDLING PARTIAL OR MISSING INFORMATION:
- If the policy explicitly defines a rule → apply it logically.
- If the policy is silent or ambiguous → clearly state the limitation.
- Absence of a policy should be treated as “not provided”.

OUTPUT FORMAT:
- Start with a 1-line direct answer.
- Then give up to 2–3 bullet points ONLY if needed.
- If a bullet list is not needed, respond with only the direct answer.

ONLY respond with:
"I could not find this information in the provided documents."
IF AND ONLY IF:
- The question is about a company-specific rule or entitlement,
- AND the context contains no relevant policy text,
- AND the answer cannot be reasonably inferred from any provided rule.

Context:
{context}

Question:
{query}

Answer:

""".strip()

    print(f"\n🧠 Prompt tokens: {count_tokens(prompt)}")
    # print(f"prompt: {prompt}")
    print("\n✅ Answer (streaming):\n")

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
# -----------------------------
# RAG ANSWERING
# -----------------------------
@timer_decorator
def answer_question(query, index, metadata):
    chunks = retrieve(query, index, metadata)

    if not chunks:
        return "I could not find this information in the provided documents."
    chunks = rerank_chunks(query, chunks, top_n=TOP_N_RERANK)
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
4. For general HR terminology questions, provide a short, standard definition even if not present in the documents.


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
    # print(f"🧠 Prompt to LLM: {prompt}")
    print(f"🧠 Prompt tokens: {count_tokens(prompt)}")
    return call_llm(prompt)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # index, metadata = load_store()

    print("🤖 RAG Retriever ready (type 'exit' to quit)")

    while True:
        query = input("\nAsk a question: ").strip()
        if query.lower() == "exit" or query.lower() == "q":
            break
        try:
            for token in answer_question_stream(query):
                print(token, end="", flush=True)
            print("\n")  # newline after streaming completes
        except Exception as e:
            print("\n❌ Error during LLM call:")
            print(str(e))
        # try:
        #     answer = answer_question(query, index, metadata)
        #     print("\n✅ Answer:\n")
        #     print(answer)
        # except Exception as e:
        #     print("\n❌ Error during LLM call:")
        #     print(str(e))