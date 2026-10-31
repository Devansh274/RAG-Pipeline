"""All tunable settings for the RAG evaluation live here.
Edit this file to change what the pipeline does; no other file needs touching.
This module imports nothing from the project (it is the bottom of the tree)."""

from pathlib import Path

# ============================== CONFIG ==============================
# Point this at ONE benchmark json to start (smaller ones load/run faster).
BENCHMARK_PATH = Path("data/benchmarks/cuad.json")
CORPUS_ROOT = Path("data/corpus")

# How many distinct source documents to load, and how many queries to test.
# Queries are only selected if their ground-truth snippets fit within this
# many documents, so keeping both small keeps the comparison fast and cheap.
#
# Because BENCHMARK_PATH above points at cuad.json only, load_benchmark()
# walks CUAD's tests exclusively -- every selected query comes from CUAD,
# and every document it pulls in comes from each query's snippet file_paths,
# which are themselves CUAD corpus files (e.g. "cuad/<contract>.txt" under
# CORPUS_ROOT). So pointing BENCHMARK_PATH at cuad.json is sufficient to
# constrain both documents and queries to CUAD -- no change to
# load_benchmark's selection logic itself is needed.
MAX_DOCUMENTS = 200
NUM_QUERIES = 1500

USE_FIXED_DOCUMENT_SUBSET = True
FIXED_DOCUMENT_SUBDIR = "cuad"

# Chunking parameters (characters, not tokens — LegalBench-RAG's ground truth
# spans are character offsets, so keeping chunk math in characters avoids a
# token<->character conversion step). Chunking is applied once and shared
# across all models below, so the comparison isolates the embedding model.
CHUNKING_SCHEME = "structure"  # "fixed" or "structure"
FIXED_CHUNK_SIZE = 500
FIXED_CHUNK_OVERLAP = 100
STRUCTURE_MAX_CHUNK_SIZE = 800
STRUCTURE_CHUNK_OVERLAP = 100

TOP_K = 8  # how many chunks to retrieve per query
MIN_HIT_OVERLAP_FRAC = 0.5

USE_CHROMA = True                 # False = old in-memory store, re-embeds every run
CHROMA_DIR = Path("chroma_db")    # on-disk location of the database
CHROMA_REBUILD = False            # True = delete this run's collection and re-embed
CHROMA_ADD_BATCH = 500            # chunks embedded + saved per step

# --- Two-stage retrieval: (1) metadata -> document subset, (2) similarity ---
# Queries are "<document hint>; <question>". Stage 1 ranks DOCUMENTS against
# the hint (the text before the first separator) and keeps a small subset;
# stage 2 runs the usual chunk-level cosine search, using only the question
# (the text after the separator), restricted to chunks of that subset.
# If a query has no separator (or an empty side), it falls back to searching
# every document with the full query text, so it is never dropped.
# Set METADATA_RETRIEVAL_ENABLED = False to reproduce the old behaviour
# exactly (full query, whole corpus) -- handy as a baseline run.
METADATA_RETRIEVAL_ENABLED = True
METADATA_QUERY_SEPARATOR = ";"
# How documents are matched against the hint:
#   "bm25"      lexical match on file name + document header. Free, no API
#               calls, strong when the hint repeats party/policy names.
#   "embedding" cosine similarity between the hint and each document's
#               name+header, using the SAME embedding model being evaluated
#               (costs one embedding per document + one per query).
#   "hybrid"    reciprocal-rank fusion of the two above.
METADATA_METHOD = "bm25"
METADATA_TOP_DOCS = 3          # max documents kept after stage 1
METADATA_MIN_SCORE_RATIO = 0.5 # bm25 only: also drop docs scoring below this
                               # fraction of the best doc's score (0 = off)
METADATA_HEADER_CHARS = 1500   # first N chars of each doc used as its "header"
METADATA_FILENAME_WEIGHT = 3   # file-name tokens count this many times in BM25

# Models to compare. "provider" selects which embed_* function is used.
# Swap model names here if you want e.g. text-embedding-3-small or
# embed-english-v3.0 instead.
MODELS = [
    {"provider": "voyage", "name": "voyage-4-large"},
    # {"provider": "cohere", "name": "embed-v4.0"},
    # {"provider": "local", "name": "BAAI/bge-large-en-v1.5"},
]

ANSWER_PREVIEW_CHARS = 100  # how much of each retrieved chunk to print

# --- LLM-based reranking (optional second stage on top of vector search) ---
# When enabled, retrieval per query becomes two-stage:
#   1. Plain cosine similarity (the existing vector store search) pulls a
#      larger candidate pool of RERANK_CANDIDATE_POOL chunks.
#   2. An LLM (Claude) is shown the query plus every candidate and scores
#      each one's relevance directly -- not via embedding distance -- and
#      the pool is re-sorted by that score before taking the top TOP_K.
# This is reported as a SEPARATE row per embedding model ("<model> + LLM
# rerank") alongside the plain cosine-only row, so the summary table shows
# directly whether reranking helped, hurt, or made no measurable difference
# for each embedding model on this corpus -- rather than assuming it does.
RERANK_ENABLED = False
RERANK_PROVIDER = "gemini"          # "anthropic" or "gemini"
RERANK_MODEL = "gemini-3.5-flash-lite"   # requires GEMINI_API_KEY to be set
                                     # (anthropic alternative: "claude-sonnet-5",
                                     # requires ANTHROPIC_API_KEY instead)
RERANK_CANDIDATE_POOL = 20          # how many cosine-ranked chunks to hand to the LLM
RERANK_MAX_CHARS_PER_CHUNK = 650    # truncate chunk text in the rerank prompt (cost control)
# ======================================================================
