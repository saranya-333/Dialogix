import fitz  # PyMuPDF
import re
import uuid
import os

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PDF_FOLDER = "Policy_Documents_iAgami"

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "hr_policies"

CHUNKS_DEBUG_FILE = "chunks_debug.txt"


embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True}
)

# -----------------------------
# TEXT NORMALIZATION
# -----------------------------
def normalize_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"[➢•◦▪]+", " ", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()

# -----------------------------
# HEADING DETECTION
# -----------------------------
def is_heading(line: str):
    clean = normalize_line(line)

    match = re.search(
        r"\b(\d+(\.\d+)*)\.\s+([A-Za-z][A-Za-z\s()/\-]{3,})",
        clean
    )
    if match:
        return True, f"{match.group(1)}. {match.group(3).strip()}"

    if clean.isupper() and len(clean.split()) <= 7:
        return True, clean

    return False, None

# -----------------------------
# STRUCTURAL CHUNKING
# (EMBED DOCUMENT + SECTION TITLE)
# -----------------------------
def extract_structural_chunks(pdf_path, document_title):
    doc = fitz.open(pdf_path)
    documents = []

    current_section = "INTRODUCTION"
    buffer = []

    for page_no, page in enumerate(doc, start=1):
        text = page.get_text()

        for raw_line in text.split("\n"):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            is_hdr, title = is_heading(raw_line)

            if is_hdr:
                if buffer:
                    content_text = " ".join(buffer).strip()

                    embedded_text = f"""
Document Title: {document_title}
Section Title: {current_section}
Content:
{content_text}
""".strip()

                    documents.append(
                        Document(
                            page_content=embedded_text,
                            metadata={
                                "id": str(uuid.uuid4()),
                                "document_title": document_title,
                                "section_title": current_section,
                                "page_number": page_no
                            }
                        )
                    )
                    buffer = []

                current_section = title
            else:
                buffer.append(normalize_line(raw_line))

    # Flush remaining buffer
    if buffer:
        content_text = " ".join(buffer).strip()

#         embedded_text = f"""
# Document Title: {document_title}
# Section Title: {current_section}
# Content:
# {content_text}
# """.strip()

        documents.append(
            Document(
             page_content=content_text,
                metadata={
                    "id": str(uuid.uuid4()),
                    "document_title": document_title,
                    "section_title": current_section,
                    "page_number": page_no
                }
            )
        )

    return documents

# -----------------------------
# PDF TITLE EXTRACTION
# -----------------------------
def get_pdf_title(pdf_path: str) -> str:
    """
    Extracts title from PDF by getting only the first line at the top.
    """
    doc = fitz.open(pdf_path)
    
    # Metadata first
    metadata_title = doc.metadata.get("title", None)
    if metadata_title:
        metadata_title = metadata_title.strip()
        # Clean up metadata prefixes
        if "Microsoft Word" in metadata_title:
            idx = metadata_title.find("Microsoft Word -")
            if idx >= 0:
                metadata_title = metadata_title[idx + len("Microsoft Word -"):].strip()
        elif "Adobe PDF" in metadata_title:
            idx = metadata_title.find("Adobe PDF -")
            if idx >= 0:
                metadata_title = metadata_title[idx + len("Adobe PDF -"):].strip()
        doc.close()
        return metadata_title
    
    page = doc[0]
    blocks = page.get_text("dict")["blocks"]
    spans = []
    
    # Collect all spans with position and font info
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:  # Only non-empty spans
                        spans.append({
                            "text": text,
                            "size": span["size"],
                            "bbox": span["bbox"],  # (x0, y0, x1, y1)
                        })
    
    if not spans:
        doc.close()
        return "No title found"
    
    # Sort all spans by vertical position (top to bottom)
    spans.sort(key=lambda s: s["bbox"][1])
    
    # Get only the first 1-2 lines (within 20 pixels of the first span)
    first_y = spans[0]["bbox"][1]
    first_line_spans = [s for s in spans if abs(s["bbox"][1] - first_y) <= 20]
    
    best_title = " ".join(s["text"] for s in first_line_spans).strip()
    
    # Clean up common PDF metadata prefixes
    if "Microsoft Word" in best_title:
        idx = best_title.find("Microsoft Word -")
        if idx >= 0:
            best_title = best_title[idx + len("Microsoft Word -"):].strip()
    elif "Adobe PDF" in best_title:
        idx = best_title.find("Adobe PDF -")
        if idx >= 0:
            best_title = best_title[idx + len("Adobe PDF -"):].strip()
    
    doc.close()
    return best_title
# -----------------------------
# LOAD ALL PDFs
# -----------------------------
def load_all_documents(folder_path):
    all_documents = []

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(folder_path, file_name)
        document_title = get_pdf_title(pdf_path)
        if not document_title or document_title == "No title found":
            document_title = os.path.splitext(file_name)[0]

        print(f" Processing: {file_name}")

        docs = extract_structural_chunks(pdf_path, document_title)
        all_documents.extend(docs)

        print(f"   ↳ Chunks created: {len(docs)}")

    return all_documents

# -----------------------------
# SAVE DEBUG FILE
# -----------------------------
def save_chunks_debug_file(documents):
    with open(CHUNKS_DEBUG_FILE, "w", encoding="utf-8") as f:
        for i, d in enumerate(documents, start=1):
            f.write("=" * 80 + "\n")
            f.write(f"CHUNK #{i}\n")
            f.write(f"Document : {d.metadata['document_title']}\n")
            f.write(f"Section  : {d.metadata['section_title']}\n")
            f.write(f"Page     : {d.metadata['page_number']}\n")
            f.write("-" * 80 + "\n")
            f.write(d.page_content + "\n\n")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print(" Loading all PDFs from folder...")
    documents = load_all_documents(PDF_FOLDER)

    print(f"\n Total chunks across all documents: {len(documents)}")

    print("Creating embeddings and storing in LangChain ChromaDB...")

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    

    save_chunks_debug_file(documents)

    print("\n Embedding completed successfully")
    print(f" Chroma directory : {CHROMA_DIR}")
    print(f" Collection name : {COLLECTION_NAME}")
    print(f" Chunk debug file: {CHUNKS_DEBUG_FILE}")
