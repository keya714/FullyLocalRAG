#!/usr/bin/env python3
import os
import yaml
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def read_config(config_path: str = "config.yml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_pdfs(pdf_dir: Path) -> List[Path]:
    return sorted([p for p in pdf_dir.rglob("*.pdf") if p.is_file()])


def load_docs_from_pdfs(pdf_paths: List[Path]):
    docs = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf_path))
            file_docs = loader.load()
            for d in file_docs:
                d.metadata = d.metadata or {}
                d.metadata.setdefault("source", str(pdf_path.resolve()))
            docs.extend(file_docs)
        except Exception as e:
            print(f"[WARN] Skipping {pdf_path}: {e}")
    return docs


def build_text_splitter(chunk_size: int, chunk_overlap: int):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )


def get_embeddings(model_name: str, device: str):
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs=encode_kwargs,
        cache_folder=os.environ.get("HF_HOME", None),
    )


def main():
    # --- Load configuration ---
    config = read_config("config.yml")

    pdf_dir = Path(config["paths"]["pdf_dir"]).expanduser().resolve()
    index_dir = Path(config["paths"]["index_dir"]).expanduser().resolve()
    model_name = config["embedding"]["model_name"]
    device = config["embedding"]["device"]
    chunk_size = int(config["chunking"]["chunk_size"])
    chunk_overlap = int(config["chunking"]["chunk_overlap"])

    index_dir.mkdir(parents=True, exist_ok=True)

    # --- Discover PDFs ---
    pdf_paths = discover_pdfs(pdf_dir)
    if not pdf_paths:
        print(f"[ERROR] No PDFs found in {pdf_dir}")
        return
    print(f"[INFO] Found {len(pdf_paths)} PDFs. Loading...")

    raw_docs = load_docs_from_pdfs(pdf_paths)
    if not raw_docs:
        print("[ERROR] No documents loaded. Aborting.")
        return

    # --- Split into chunks ---
    print(f"[INFO] Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
    splitter = build_text_splitter(chunk_size, chunk_overlap)
    docs = splitter.split_documents(raw_docs)
    print(f"[INFO] Created {len(docs)} text chunks.")

    # --- Build embeddings ---
    print(f"[INFO] Loading embedding model: {model_name} on {device}")
    embeddings = get_embeddings(model_name, device)

    # --- Build FAISS index ---
    print("[INFO] Building FAISS index...")
    vector_store = FAISS.from_documents(docs, embeddings)

    # --- Save index ---
    print(f"[INFO] Saving FAISS index to {index_dir}")
    vector_store.save_local(str(index_dir))

    # --- Manifest for reproducibility ---
    manifest = index_dir / "manifest.txt"
    manifest.write_text(
        f"pdf_dir={pdf_dir}\n"
        f"index_dir={index_dir}\n"
        f"model_name={model_name}\n"
        f"device={device}\n"
        f"chunk_size={chunk_size}\n"
        f"chunk_overlap={chunk_overlap}\n"
    )

    print("[DONE] Ingestion complete.")


if __name__ == "__main__":
    main()
