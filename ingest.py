#!/usr/bin/env python3
import os
import re
import yaml
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rich.console import Console
from rich.progress import track

console = Console()

# ---------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------

def read_config(path: str = "config.yml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def discover_pdfs(pdf_dir: Path) -> List[Path]:
    return sorted([p for p in pdf_dir.rglob("*.pdf") if p.is_file()])

# ---------------------------------------------------------------------
# Text cleaning utility
# ---------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Cleans text by:
    - Removing special characters (except punctuation and numbers)
    - Collapsing multiple whitespaces
    - Stripping leading/trailing spaces
    """
    # Remove non-alphanumeric/special chars but keep punctuation and spaces
    text = re.sub(r"[^a-zA-Z0-9.,;:!?()\-–—\s]", " ", text)
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    # Trim
    return text.strip()

# ---------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------

def load_docs_from_pdfs(pdf_paths: List[Path]) -> List[Document]:
    docs: List[Document] = []
    for pdf_path in track(pdf_paths, description="Loading PDFs"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            file_docs = loader.load()
            for d in file_docs:
                d.metadata = d.metadata or {}
                d.metadata.setdefault("source", str(pdf_path.resolve()))
                d.page_content = clean_text(d.page_content)
            docs.extend(file_docs)
        except Exception as e:
            console.print(f"[yellow]WARN[/] Skipping {pdf_path}: {e}")
    return docs

# ---------------------------------------------------------------------
# Main ingestion flow
# ---------------------------------------------------------------------

def main():
    cfg = read_config("config.yml")

    pdf_dir = Path(cfg["paths"]["pdf_dir"]).expanduser().resolve()
    index_dir = Path(cfg["paths"]["index_dir"]).expanduser().resolve()
    model_name = cfg["embedding"]["model_name"]
    device = cfg["embedding"]["device"]
    chunk_size = int(cfg["chunking"]["chunk_size"])
    chunk_overlap = int(cfg["chunking"]["chunk_overlap"])

    index_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = discover_pdfs(pdf_dir)
    if not pdf_paths:
        console.print(f"[red]ERROR[/] No PDFs found in {pdf_dir}")
        return
    console.print(f"[green]INFO[/] Found {len(pdf_paths)} PDFs. Loading…")

    raw_docs = load_docs_from_pdfs(pdf_paths)
    if not raw_docs:
        console.print("[red]ERROR[/] No documents loaded. Aborting.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    docs = splitter.split_documents(raw_docs)
    console.print(f"[green]INFO[/] Created {len(docs)} chunks after cleaning.")

    console.print(f"[green]INFO[/] Loading embeddings: {model_name} on {device}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=os.environ.get("HF_HOME", None),
    )

    console.print("[green]INFO[/] Building FAISS index…")
    vector_store = FAISS.from_documents(docs, embeddings)

    console.print(f"[green]INFO[/] Saving index to {index_dir}")
    vector_store.save_local(str(index_dir))

    manifest = index_dir / "manifest.txt"
    manifest.write_text(
        f"pdf_dir={pdf_dir}\n"
        f"index_dir={index_dir}\n"
        f"model_name={model_name}\n"
        f"device={device}\n"
        f"chunk_size={chunk_size}\n"
        f"chunk_overlap={chunk_overlap}\n"
    )
    console.print("[bold green]DONE[/] Ingestion complete.")

if __name__ == "__main__":
    main()
