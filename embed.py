"""
embed.py
--------
Run this script ONCE to build and save the FAISS vectorstore to disk.
Uses the FREE local HuggingFace model:
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
No API key required. The model is downloaded automatically on first run
and cached locally (~120 MB).

Usage:
    pip install sentence-transformers langchain-community faiss-cpu PyPDF2
    python embed.py
"""

import os
import json
import glob
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Folder that contains your PDFs and the JSON chunks file
SOURCES_FOLDER = r"C:\pooshe kaar\Health-Information-RAG-Chatbot\sources"

# JSON file produced by your scraper
JSON_CHUNKS_FILE = os.path.join(SOURCES_FOLDER, "diabetes_chunks.json")

# Where to save the FAISS index (folder will be created if it doesn't exist)
FAISS_INDEX_PATH = os.path.join(SOURCES_FOLDER, "faiss_index")

# HuggingFace model — free, local, multilingual (Finnish included)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Batch size for FAISS merging (no token-limit issue with local models;
# larger batches = faster, but uses more RAM)
BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_pdfs_from_folder(folder_path: str):
    pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
    return [open(p, "rb") for p in pdf_paths]


def get_pdf_text(pdf_docs) -> str:
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text


def get_text_chunks(text: str):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(text)


def load_json_chunks(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_embeddings_in_batches(texts, metadatas, embeddings, batch_size=500):
    """
    Build a FAISS vectorstore by processing texts in batches and merging.
    With a local HuggingFace model there are no API token limits, so you can
    use large batches (500-1000) for speed.
    """
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size
    vectorstore = None

    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_meta  = metadatas[i : i + batch_size]
        batch_num   = i // batch_size + 1
        pct         = batch_num / total_batches * 100
        print(f"  Batch {batch_num}/{total_batches} ({pct:.1f}%)  — {len(batch_texts)} chunks")

        batch_vs = FAISS.from_texts(
            texts=batch_texts, embedding=embeddings, metadatas=batch_meta
        )

        if vectorstore is None:
            vectorstore = batch_vs
        else:
            vectorstore.merge_from(batch_vs)

    print(f"[OK] All {total_batches} batches processed successfully")
    return vectorstore


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load the local embedding model (downloaded once, then cached)
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print("  (First run will download ~120 MB — subsequent runs use the cache)\n")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},   # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    all_chunks = []

    # ── PDFs ──────────────────────────────────────────────────────────────
    print("Processing PDF files...")
    pdf_files = load_pdfs_from_folder(SOURCES_FOLDER)
    if pdf_files:
        pdf_text = get_pdf_text(pdf_files)
        print(f"  Total PDF text: {len(pdf_text):,} characters")
        pdf_chunks = get_text_chunks(pdf_text)
        print(f"  Created {len(pdf_chunks)} PDF chunks")
        for chunk in pdf_chunks:
            all_chunks.append({"text": chunk, "metadata": {"source": "PDF"}})
    else:
        print("  No PDF files found — skipping.")

    # ── JSON ──────────────────────────────────────────────────────────────
    if os.path.exists(JSON_CHUNKS_FILE):
        print("\nProcessing JSON chunks...")
        json_chunks = load_json_chunks(JSON_CHUNKS_FILE)
        print(f"  Loaded {len(json_chunks)} JSON chunks")
        for item in json_chunks:
            all_chunks.append(
                {"text": item["text"], "metadata": {"source": item["source"]}}
            )
    else:
        print(f"\n[WARNING] JSON file not found: {JSON_CHUNKS_FILE} — skipping.")

    if not all_chunks:
        raise RuntimeError("No content found to embed. Check your folder paths.")

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    texts     = [x["text"]     for x in all_chunks]
    metadatas = [x["metadata"] for x in all_chunks]

    # ── Embed & build FAISS ───────────────────────────────────────────────
    print("\nBuilding FAISS vectorstore (this may take a few minutes on CPU)...")
    vectorstore = create_embeddings_in_batches(
        texts, metadatas, embeddings, batch_size=BATCH_SIZE
    )

    # ── Save to disk ──────────────────────────────────────────────────────
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"\n[OK] Vectorstore saved to: {FAISS_INDEX_PATH}")
    print("You can now start app.py — it will load this index without re-embedding.")


if __name__ == "__main__":
    main()
