"""Step 4 - two-stage retrieval.
Stage 1: DocumentMetadataIndex (BM25 / embedding / hybrid) picks candidate docs.
Stage 2: vector search (Chroma on disk, or LangChain's in-memory store)."""

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

import chromadb
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from config import (
    CHROMA_ADD_BATCH,
    CHROMA_DIR,
    CHROMA_REBUILD,
    CHUNKING_SCHEME,
    METADATA_FILENAME_WEIGHT,
    METADATA_HEADER_CHARS,
    METADATA_METHOD,
    METADATA_MIN_SCORE_RATIO,
    METADATA_QUERY_SEPARATOR,
    METADATA_TOP_DOCS,
)
from embeddings import _embed_chunks_raw


#   Stage 1 -- metadata retrieval: DocumentMetadataIndex ranks documents
#     against the hint part of the query and returns a small subset.
#   Stage 2 -- similarity search: plain top-k cosine search via a LangChain
#     VectorStore, using the question part of the query, scoped to the stage-1
#     subset via the store's `filter` argument.
# Each model gets its own InMemoryVectorStore, built once (all chunks embedded
# a single time via add_texts, using the same ProviderEmbeddings adapter as
# before) and reused across all queries in the run. The ranking itself is the
# vector store's own top-k similarity search -- no overlap-aware re-ranking.
#
# Unlike an earlier version of this script, the subset is NOT derived from
# the ground-truth spans (that leaked the answer). It comes only from the
# query's own hint text, so stage-1 misses are real and are reported.

_METADATA_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "for", "to", "in", "on", "by", "with",
    "between", "among", "from", "at", "as", "is", "are", "s",
    "consider", "regarding", "about", "under", "per", "its", "their",
}


def split_query(query, sep=METADATA_QUERY_SEPARATOR):
    """Split "<hint>; <question>" on the FIRST separator only (the question
    itself may contain more ';'). Returns (hint, question). If there is no
    separator or either side is empty, returns (None, full_query) so the
    caller can fall back to unfiltered search with the whole query."""
    hint, found, question = query.partition(sep)
    hint, question = hint.strip(), question.strip()
    if not found or not hint or not question:
        return None, query.strip()
    return hint, question


def _tokenize(text):
    """Lowercase word tokens; splits CamelCase ("AcmeHoldingsInc" ->
    acme/holdings/inc) and drops stopwords."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return [t for t in re.findall(r'[a-z0-9]+', text.lower())
            if t not in _METADATA_STOPWORDS]


class _BM25:
    """Minimal Okapi BM25 over a small list of token lists (no extra
    dependency). IDF uses the non-negative Lucene form."""

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(t) for t in docs_tokens]
        self.dl = np.array([len(t) for t in docs_tokens], dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if len(self.dl) and self.dl.mean() > 0 else 1.0
        n = len(docs_tokens)
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query_tokens):
        out = np.zeros(len(self.tfs))
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / norm
        return out


class DocumentMetadataIndex:
    """Stage 1. Document-level 'metadata' = file name + the first
    METADATA_HEADER_CHARS characters (title / parties / preamble for
    contracts, policy name for privacy policies). Built once per run."""

    def __init__(self, documents):
        self.doc_ids = list(documents)
        self.texts, tokens = [], []
        for fp in self.doc_ids:
            name = re.sub(r'[_\-]+', ' ', Path(fp).stem)
            header = documents[fp][:METADATA_HEADER_CHARS]
            self.texts.append(f"{name}\n{header}")
            tokens.append(_tokenize(name) * METADATA_FILENAME_WEIGHT + _tokenize(header))
        self.bm25 = _BM25(tokens)
        self.doc_vecs = None

    def embed(self, provider, model_name):
        """Only needed for the "embedding"/"hybrid" methods."""
        self.doc_vecs = _embed_chunks_raw(
            provider, model_name, [{"text": t} for t in self.texts])

    def select(self, hint, hint_vec=None):
        """Returns (list_of_file_paths, note). `note` is None for a normal
        filtered result, or a short reason when it fell back to ALL documents."""
        all_docs = list(self.doc_ids)
        if hint is None:
            return all_docs, "no '<hint>;' prefix in query -> searched all documents"

        method = METADATA_METHOD
        top = max(1, min(METADATA_TOP_DOCS, len(all_docs)))
        use_bm25 = method in ("bm25", "hybrid")
        use_emb = method in ("embedding", "hybrid")
        if method not in ("bm25", "embedding", "hybrid"):
            raise ValueError(f"Unknown METADATA_METHOD: {method!r}")

        bm = self.bm25.scores(_tokenize(hint)) if use_bm25 else None
        if use_bm25 and bm.max() <= 0:
            if method == "bm25":
                return all_docs, "hint shares no terms with any document -> searched all documents"
            use_bm25 = False  # hybrid degrades to embedding-only

        if method == "bm25":
            order = np.argsort(-bm, kind="stable")
            best = bm[order[0]]
            keep = [i for i in order[:top] if bm[i] >= METADATA_MIN_SCORE_RATIO * best]
            return [self.doc_ids[i] for i in keep], None

        if self.doc_vecs is None or hint_vec is None:
            raise RuntimeError("embedding/hybrid metadata retrieval needs doc and hint vectors")
        fused = np.zeros(len(all_docs))
        for s in ((bm if use_bm25 else None), self.doc_vecs @ hint_vec):
            if s is None:
                continue
            ranks = np.empty(len(s), dtype=int)
            ranks[np.argsort(-s, kind="stable")] = np.arange(len(s))
            fused += 1.0 / (60 + ranks)  # reciprocal-rank fusion
        order = np.argsort(-fused, kind="stable")
        return [self.doc_ids[i] for i in order[:top]], None


def build_vector_store(embeddings_model, chunks_by_doc):
    """Embeds every chunk across all documents once and returns an
    InMemoryVectorStore over them, with each chunk's file_path/start/end
    carried as metadata so results can be mapped back to source offsets."""
    vector_store = InMemoryVectorStore(embeddings_model)
    all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]
    vector_store.add_texts(
        texts=[c["text"] for c in all_chunks],
        metadatas=[
            {"file_path": c["file_path"], "start": c["start"], "end": c["end"]}
            for c in all_chunks
        ],
    )
    return vector_store


def _doc_to_chunk(doc):
    return {
        "text": doc.page_content,
        "file_path": doc.metadata["file_path"],
        "start": doc.metadata["start"],
        "end": doc.metadata["end"],
    }

def _chroma_collection_name(provider, model_name, all_chunks):
    """One collection per (model, chunking scheme, exact chunk content). Any
    change to the model, the chunker, or the documents yields a new name, so
    stale vectors are never reused."""
    h = hashlib.sha256()
    h.update(f"{provider}|{model_name}|{CHUNKING_SCHEME}".encode())
    for c in all_chunks:
        h.update(f"{c['file_path']}|{c['start']}|{c['end']}|".encode())
        h.update(c["text"].encode("utf-8"))
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{provider}_{model_name}")[:60]
    return f"{base}_{h.hexdigest()[:16]}"


def build_chroma_collection(provider, model_name, chunks_by_doc):
    """Opens (or creates) the persistent collection and embeds ONLY the chunks
    it doesn't already contain. Because ids are deterministic and each batch
    is saved as it goes, a run that dies mid-way (rate limit etc.) resumes
    where it stopped."""
    all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]
    name = _chroma_collection_name(provider, model_name, all_chunks)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if CHROMA_REBUILD:
        try:
            client.delete_collection(name)
        except Exception:
            pass  # didn't exist yet
    collection = client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"})

    ids = [f"{c['file_path']}|{c['start']}|{c['end']}" for c in all_chunks]
    existing = set(collection.get(include=[])["ids"])
    todo = [i for i, id_ in enumerate(ids) if id_ not in existing]
    print(f"  Chroma collection '{name}': {len(existing)} chunks stored, "
          f"{len(todo)} to embed.")
    for s in range(0, len(todo), CHROMA_ADD_BATCH):
        part = todo[s:s + CHROMA_ADD_BATCH]
        vecs = _embed_chunks_raw(provider, model_name, [all_chunks[i] for i in part])
        collection.add(
            ids=[ids[i] for i in part],
            embeddings=vecs.tolist(),
            documents=[all_chunks[i]["text"] for i in part],
            metadatas=[{"file_path": all_chunks[i]["file_path"],
                        "start": all_chunks[i]["start"],
                        "end": all_chunks[i]["end"]} for i in part],
        )
        print(f"    embedded + saved {min(s + CHROMA_ADD_BATCH, len(todo))}/{len(todo)}")
    return collection


def chroma_search(collection, query_vec, k, doc_subset=None):
    """Top-k chunks by cosine similarity, optionally restricted to documents.
    Returns [(Document, similarity)], the same shape the old code consumed."""
    where = {"file_path": {"$in": list(doc_subset)}} if doc_subset else None
    res = collection.query(
        query_embeddings=[query_vec.tolist()], n_results=k, where=where,
        include=["documents", "metadatas", "distances"])
    return [(Document(page_content=t, metadata=m), 1.0 - d)   # cosine distance -> similarity
            for t, m, d in zip(res["documents"][0], res["metadatas"][0],
                               res["distances"][0])]
