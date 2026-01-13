# embedder.py

import fitz  # PyMuPDF
import re
import uuid
import json
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
PDF_FOLDER = r"C:\Users\logesh\Downloads\Policy_Documents_–_iAgami"

MODEL_NAME = "all-MiniLM-L6-v2"   # FREE local model
EMBED_DIM = 384

INDEX_FILE = "vector.index"
META_FILE = "metadata.json"
CHUNKS_DEBUG_FILE = "chunks_debug.txt"   # 👈 NEW

model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# HEADING DETECTION
# -----------------------------
def is_heading(line: str):
    line = line.strip()

    # Numbered headings: 1, 1.1, 2.3.4 Title
    if re.match(r"^\d+(\.\d+)*\s+.+", line):
        parts = line.split(maxsplit=1)
        return True, parts[1] if len(parts) > 1 else line

    # ALL CAPS headings ending with :
    if line.isupper() and line.endswith(":") and len(line.split()) <= 7:
        return True, line.rstrip(":")

    return False, None

# -----------------------------
# STRUCTURAL CHUNKING (PER PDF)
# -----------------------------
def extract_structural_chunks(pdf_path, document_title):
    doc = fitz.open(pdf_path)
    chunks = []

    current_section = "INTRODUCTION"
    buffer = []

    for page_no, page in enumerate(doc, start=1):
        text = page.get_text()

        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            is_hdr, title = is_heading(line)

            if is_hdr:
                if buffer:
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "document_title": document_title,
                        "section_title": current_section,
                        "page_number": page_no,
                        "content": " ".join(buffer).strip()
                    })
                    buffer = []

                current_section = title
            else:
                buffer.append(line)

    if buffer:
        chunks.append({
            "id": str(uuid.uuid4()),
            "document_title": document_title,
            "section_title": current_section,
            "page_number": page_no,
            "content": " ".join(buffer).strip()
        })

    return chunks

# -----------------------------
# LOAD ALL PDFs FROM FOLDER
# -----------------------------
def load_all_documents(folder_path):
    all_chunks = []

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(folder_path, file_name)
        document_title = os.path.splitext(file_name)[0]

        print(f"📄 Processing: {file_name}")

        chunks = extract_structural_chunks(pdf_path, document_title)
        all_chunks.extend(chunks)

        print(f"   ↳ Chunks created: {len(chunks)}")

    return all_chunks

# -----------------------------
# BUILD VECTOR INDEX
# -----------------------------
def build_index(chunks):
    index = faiss.IndexFlatIP(EMBED_DIM)

    texts = [c["content"] for c in chunks]

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    index.add(vectors.astype("float32"))
    return index, chunks

# -----------------------------
# SAVE INDEX + METADATA
# -----------------------------
def save(index, metadata):
    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

# -----------------------------
# SAVE DEBUG CHUNK FILE (NEW)
# -----------------------------
def save_chunks_debug_file(chunks):
    with open(CHUNKS_DEBUG_FILE, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks, start=1):
            f.write(f"{'='*80}\n")
            f.write(f"CHUNK #{i}\n")
            f.write(f"Document : {c['document_title']}\n")
            f.write(f"Section  : {c['section_title']}\n")
            f.write(f"Page     : {c['page_number']}\n")
            f.write(f"{'-'*80}\n")
            f.write(c["content"])
            f.write("\n\n")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("📂 Loading all PDFs from folder...")
    all_chunks = load_all_documents(PDF_FOLDER)

    print(f"\n🧩 Total chunks across all documents: {len(all_chunks)}")

    print("🧠 Creating embeddings locally (no API, no cost)...")
    index, metadata = build_index(all_chunks)

    save(index, metadata)
    save_chunks_debug_file(all_chunks)   # 👈 NEW

    print("\n✅ Embedding completed successfully")
    print(f"📁 Vector index saved as : {INDEX_FILE}")
    print(f"📁 Metadata saved as     : {META_FILE}")
    print(f"📁 Chunk debug file     : {CHUNKS_DEBUG_FILE}")
