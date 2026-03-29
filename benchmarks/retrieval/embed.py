#!/usr/bin/env python3
"""
Generate reproducible embeddings for the retrieval benchmark corpus and queries.

Preferred backend:
    sentence-transformers/all-MiniLM-L6-v2

Fallback backend:
    scikit-learn TF-IDF + TruncatedSVD
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "corpus.jsonl"
QUERIES_PATH = ROOT / "queries.jsonl"
OUTPUT_DIR = ROOT / "results"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BACKEND = "auto"
FALLBACK_MODEL = "tfidf-svd"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_doc_text(record: dict[str, Any]) -> str:
    tags = " ".join(record.get("tags", []))
    return f"{record['title']}\n{record['body']}\n{tags}".strip()


def build_query_text(record: dict[str, Any]) -> str:
    return record["query"]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def save_metadata(
    output_dir: Path,
    backend: str,
    model_name: str,
    doc_records: list[dict[str, Any]],
    query_records: list[dict[str, Any]],
    dimension: int,
) -> None:
    metadata = {
        "backend": backend,
        "model_name": model_name,
        "document_count": len(doc_records),
        "query_count": len(query_records),
        "dimension": dimension,
        "documents_file": "doc_embeddings.npy",
        "queries_file": "query_embeddings.npy",
        "document_ids_file": "doc_ids.json",
        "query_ids_file": "query_ids.json",
    }
    (output_dir / "embedding_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def encode_with_sentence_transformers(
    model_name: str,
    doc_texts: list[str],
    query_texts: list[str],
) -> tuple[np.ndarray, np.ndarray, str]:
    from sentence_transformers import SentenceTransformer

    print(f"Loading sentence-transformers model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Embedding {len(doc_texts)} corpus documents with sentence-transformers...")
    doc_embeddings = model.encode(
        doc_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print(f"Embedding {len(query_texts)} queries with sentence-transformers...")
    query_embeddings = model.encode(
        query_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    return doc_embeddings, query_embeddings, model_name


def encode_with_tfidf_svd(
    doc_texts: list[str],
    query_texts: list[str],
    max_features: int,
    svd_dim: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    print("Falling back to TF-IDF + TruncatedSVD embeddings...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
    )
    doc_matrix = vectorizer.fit_transform(doc_texts)
    query_matrix = vectorizer.transform(query_texts)

    effective_dim = min(
        svd_dim,
        max(1, doc_matrix.shape[0] - 1),
        max(1, doc_matrix.shape[1] - 1),
    )
    if effective_dim < 1:
        raise SystemExit("Corpus too small to build TF-IDF + SVD fallback embeddings")

    svd = TruncatedSVD(n_components=effective_dim, random_state=42)
    doc_embeddings = svd.fit_transform(doc_matrix).astype(np.float32)
    query_embeddings = svd.transform(query_matrix).astype(np.float32)

    doc_embeddings = normalize_rows(doc_embeddings).astype(np.float32)
    query_embeddings = normalize_rows(query_embeddings).astype(np.float32)

    return doc_embeddings, query_embeddings, f"{FALLBACK_MODEL}:{effective_dim}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["auto", "sentence-transformers", "tfidf-svd"],
        default=DEFAULT_BACKEND,
        help="Embedding backend to use",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=4096,
        help="Maximum TF-IDF features for the fallback backend",
    )
    parser.add_argument(
        "--svd-dim",
        type=int,
        default=128,
        help="Target embedding dimension for the TF-IDF + SVD fallback backend",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for embedding outputs",
    )
    args = parser.parse_args()

    doc_records = load_jsonl(CORPUS_PATH)
    query_records = load_jsonl(QUERIES_PATH)

    if not doc_records:
        raise SystemExit(f"No corpus records found in {CORPUS_PATH}")
    if not query_records:
        raise SystemExit(f"No query records found in {QUERIES_PATH}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    doc_texts = [build_doc_text(record) for record in doc_records]
    query_texts = [build_query_text(record) for record in query_records]

    backend_used = args.backend
    model_name = args.model

    if args.backend in {"auto", "sentence-transformers"}:
        try:
            doc_embeddings, query_embeddings, model_name = encode_with_sentence_transformers(
                args.model,
                doc_texts,
                query_texts,
            )
            backend_used = "sentence-transformers"
        except Exception as exc:
            if args.backend == "sentence-transformers":
                raise
            print(
                "sentence-transformers backend unavailable; "
                f"falling back to TF-IDF + SVD ({exc.__class__.__name__}: {exc})"
            )
            doc_embeddings, query_embeddings, model_name = encode_with_tfidf_svd(
                doc_texts,
                query_texts,
                max_features=args.max_features,
                svd_dim=args.svd_dim,
            )
            backend_used = "tfidf-svd"
    else:
        doc_embeddings, query_embeddings, model_name = encode_with_tfidf_svd(
            doc_texts,
            query_texts,
            max_features=args.max_features,
            svd_dim=args.svd_dim,
        )
        backend_used = "tfidf-svd"

    np.save(args.output_dir / "doc_embeddings.npy", doc_embeddings)
    np.save(args.output_dir / "query_embeddings.npy", query_embeddings)

    (args.output_dir / "doc_ids.json").write_text(
        json.dumps([record["id"] for record in doc_records], indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "query_ids.json").write_text(
        json.dumps([record["query_id"] for record in query_records], indent=2),
        encoding="utf-8",
    )

    save_metadata(
        output_dir=args.output_dir,
        backend=backend_used,
        model_name=model_name,
        doc_records=doc_records,
        query_records=query_records,
        dimension=int(doc_embeddings.shape[1]),
    )

    print("Saved:")
    print(f"  - {args.output_dir / 'doc_embeddings.npy'}")
    print(f"  - {args.output_dir / 'query_embeddings.npy'}")
    print(f"  - {args.output_dir / 'doc_ids.json'}")
    print(f"  - {args.output_dir / 'query_ids.json'}")
    print(f"  - {args.output_dir / 'embedding_metadata.json'}")
    print(f"Embedding backend: {backend_used}")
    print(f"Embedding model: {model_name}")
    print(f"Embedding dimension: {doc_embeddings.shape[1]}")
    print("Done.")


if __name__ == "__main__":
    main()
