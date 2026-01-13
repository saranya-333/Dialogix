import fitz  # PyMuPDF
import re
import faiss
import uuid
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage

# -----------------------------
# CONFIG
# -----------------------------
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims, stable
LLM_MODEL = "gpt-4o-mini"

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

EMBED_DIM = 1536

# -----------------------------
# IN-MEMORY "DB"
# -----------------------------
faiss_index = faiss.IndexFlatIP(EMBED_DIM)
metadata_store = []   # keeps metadata aligned with FAISS index

# -----------------------------
# STRUCTURAL CHUNKING
# -----------------------------
def extract_structural_chunks(pdf_path: str, document_title: str):
    doc = fitz.open(pdf_path)
    chunks = []

    current_section = None
    current_clause = None
    buffer = []

    for page_no, page in enumerate(doc, start=1):
        text = page.get_text()

        for line in text.split("\n"):
            match = re.match(r"^(\d+(\.\d+)*)\s+(.*)", line)

            if match:
                if buffer:
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "document_title": document_title,
                        "section_title": current_section,
                        "clause_number": current_clause,
                        "page_number": page_no,
                        "content": " ".join(buffer).strip()
                    })
                    buffer = []

                current_clause = match.group(1)
                current_section = match.group(3)
            else:
                buffer.append(line)

    if buffer:
        chunks.append({
            "id": str(uuid.uuid4()),
            "document_title": document_title,
            "section_title": current_section,
            "clause_number": current_clause,
            "page_number": page_no,
            "content": " ".join(buffer).strip()
        })

    return chunks

# -----------------------------
# INGEST INTO VECTOR STORE
# -----------------------------
def ingest_chunks(chunks):
    for chunk in chunks:
        vector = embeddings.embed_query(chunk["content"])
        vector = np.array(vector, dtype="float32").reshape(1, -1)

        faiss.normalize_L2(vector)
        faiss_index.add(vector)
        metadata_store.append(chunk)

# -----------------------------
# RETRIEVAL
# -----------------------------
def retrieve(query: str, top_k=5):
    q_vec = embeddings.embed_query(query)
    q_vec = np.array(q_vec, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_vec)

    scores, indices = faiss_index.search(q_vec, top_k)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(metadata_store[idx])

    return results

# -----------------------------
# ANSWERING (RAG)
# -----------------------------
def answer_question(query: str):
    retrieved_chunks = retrieve(query)

    if not retrieved_chunks:
        return "I could not find this information in the provided documents."

    context = "\n\n".join(
        f"[{c['document_title']} | {c['section_title']} ({c['clause_number']}) | Page {c['page_number']}]\n{c['content']}"
        for c in retrieved_chunks
    )

    prompt = f"""
You are a knowledge assistant.

Answer ONLY using the context below.
Do NOT infer or assume.
If the answer is not present, say:
"I could not find this information in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    pdf_path = "sample.pdf"   # put your PDF here
    document_title = "Sample Document"

    print("Ingesting document...")
    chunks = extract_structural_chunks(pdf_path, document_title)
    ingest_chunks(chunks)
    print(f"Ingested {len(chunks)} chunks")

    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("\n", answer_question(q))
