import json
import requests
import tiktoken
import time
from functools import wraps

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from prompt import rewriter_prompt,answer_prompt

MODEL_NAME = "all-MiniLM-L6-v2"

# ========================================
# EXECUTION TIME DECORATOR
# ========================================
def timer_decorator(func):
    """Decorator to measure and print the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n⏱️  {func.__name__} execution time: {elapsed_time:.4f} seconds\n")
        return result
    return wrapper


TOP_K = 20               # retrieval candidates
RERANK_TOP_N = 7        # chunks sent to LLM (rank-based)
CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "hr_policies"

LLM_URL = "http://aigateway.iagami.com:4000/v1/chat/completions"
LLM_MODEL = "oci/google.gemini-2.5-flash-lite"
LLM_API_KEY = "sk-aphZ6iCssH6F97y_wj_3Hw"  

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True}
)

# -----------------------------
# LOAD CHROMA VECTOR STORE
# -----------------------------
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}  # use cosine similarity
)


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rewrite_query_ai(query: str) -> str:
    rewrite_prompt =f"""
    {rewriter_prompt}
    
    
User query:
{query}

Rewritten query:
""".strip()

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": rewrite_prompt}
        ],
        "temperature": 0
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }

    try:
        response = requests.post(
            LLM_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        rewritten = response.json()["choices"][0]["message"]["content"].strip()
        return rewritten if rewritten else query
    except Exception:
        return query




def is_overview_query(query: str) -> bool:
    overview_keywords = [
        "explain",
        "overview",
        "summary",
        "tell me about",
        "describe"
    ]
    return any(k in query.lower() for k in overview_keywords)

# -----------------------------
# RETRIEVAL (NO FILTERING)
# -----------------------------
def retrieve(query: str):
    # 🔹 OVERVIEW QUERY (STRUCTURAL)
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

    print("\n========= RETRIEVED CHUNKS (NO FILTERING) =========\n")

    chunks = []
    for idx, (d, score) in enumerate(docs_with_scores, start=1):
        print(f"CHUNK #{idx}")
        print(f"Vector Score : {score:.4f} (distance)")
        print(f"Document     : {d.metadata.get('document_title')}")
        print(f"Section      : {d.metadata.get('section_title')}")
        print(f"Page         : {d.metadata.get('page_number')}")
        print("-" * 70)
        print(d.page_content)
        print("-" * 70 + "\n")

        chunks.append({
            "document_title": d.metadata["document_title"],
            "section_title": d.metadata["section_title"],
            "page_number": d.metadata["page_number"],
            "content": d.page_content,
            "vector_score": score
        })

    print("==================================================\n")
    return chunks

# -----------------------------
# RERANK CHUNKS (RANK-BASED ONLY)
# -----------------------------
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
# TOKEN COUNT
# -----------------------------
def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# -----------------------------
# LLM CALL
# -----------------------------
def call_llm(prompt: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 1
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
    return response.json()["choices"][0]["message"]["content"]

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
# RAG ANSWERING (PROMPT UNCHANGED)
# -----------------------------
@timer_decorator
def answer_question(query: str):
    rewritten_query = rewrite_query_ai(query)
    print(f"\n Rewritten Query: {rewritten_query}\n")
    chunks = retrieve(rewritten_query)

    if not chunks:
        return "I could not find this information in the provided documents."


    chunks = rerank_chunks(rewritten_query, chunks, RERANK_TOP_N)

    print("\n========= RERANKED CHUNKS (SENT TO LLM) =========\n")
    for i, c in enumerate(chunks, start=1):
        print(f"CHUNK #{i}")
        print(f"Rerank Score : {c['rerank_score']:.4f}")
        print(f"Document     : {c['document_title']}")
        print(f"Section      : {c['section_title']}")
        print(f"Page         : {c['page_number']}")
        print("-" * 70)
        print(c["content"])
        print("-" * 70 + "\n")

    context = "\n\n".join(
        f"[Document: {c['document_title']} | "
        f"Section: {c['section_title']} | "
        f"Page: {c['page_number']}]\n{c['content']}"
        for c in chunks
    )

    prompt = f"""
     {answer_prompt}

Context:
{context}

Question:
{rewritten_query}

Answer:
""".strip()

    print(f"\n🧠 Prompt tokens: {count_tokens(prompt)}")
    print("\n✅ Answer (streaming):\n")

    # Stream tokens
    for chunk in call_llm_stream(prompt):
        yield chunk
    
    # print(f" Prompt tokens: {count_tokens(prompt)}")
    # return call_llm(prompt)
# -----------------------------
# MAIN LOOP
# -----------------------------
if __name__ == "__main__":
    print(" LangChain + Chroma RAG Retriever with Cross-Encoder Reranker ready")

    while True:
        query = input("\nAsk a question: ").strip()
        if query.lower() in {"exit", "q"}:
            break
        try:
            for token in answer_question(query):
                print(token, end="", flush=True)
            print("\n")  # newline after streaming completes
        except Exception as e:
            print("\n❌ Error during LLM call:")
            print(str(e))
        # try:
        #     answer = answer_question(query)
        #     print("\n Answer:\n")
        #     print(answer)
        # except Exception as e:
        #     print("\n Error:")
        #     print(str(e))
