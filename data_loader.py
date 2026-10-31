"""Step 1 - load the benchmark queries and the source documents they reference."""

import json
import random


def load_benchmark(benchmark_path, corpus_root, max_documents, num_queries, shuffle_seed=0):
    """Walks the benchmark's tests in randomized order, greedily keeping a
    query only if it doesn't push the running set of distinct source
    documents past max_documents. Stops once num_queries queries are collected.

    The benchmark file groups all queries for one document together before
    moving to the next, so walking it in original order exhausts num_queries
    within the first document or two and MAX_DOCUMENTS never binds. Shuffling
    with a fixed seed spreads document coverage while staying reproducible.

    Adjust the key names below (test['query'], snippet['file_path'],
    snippet['span']) if your downloaded json uses different field names —
    print(raw[0]) after loading to check before running the full thing.
    """
    raw = json.loads(benchmark_path.read_text(encoding="utf-8"))
    tests = raw["tests"] if "tests" in raw else raw
    tests = list(tests)
    random.Random(shuffle_seed).shuffle(tests)

    selected_docs = []      # unique file_paths, in first-seen (post-shuffle) order
    selected_queries = []   # [{"query": str, "spans": [{"file_path","start","end"}]}]
    missing_files = set()   # corpus files referenced by the benchmark but absent on disk

    for test in tests:
        if len(selected_queries) >= num_queries:
            break
        snippet_docs = {s["file_path"] for s in test["snippets"]}
        new_docs = snippet_docs - set(selected_docs)
        if len(selected_docs) + len(new_docs) > max_documents:
            continue  # would need more documents than we're allowed — skip
        unavailable = {d for d in new_docs if not (corpus_root / d).exists()}
        if unavailable:
            # A referenced corpus file doesn't exist (missing download, or a
            # filename mismatch — e.g. a colon in the benchmark's file_path
            # that got sanitized during extraction). Skip this test entirely
            # rather than counting the doc toward the budget and leaving a
            # query pointing at a document that never gets loaded — that
            # would KeyError deep in the retrieval loop instead of here.
            missing_files |= unavailable
            continue
        selected_docs.extend(sorted(new_docs))
        spans = [{"file_path": s["file_path"], "start": s["span"][0], "end": s["span"][1]}
                 for s in test["snippets"]]
        selected_queries.append({"query": test["query"], "spans": spans})

    for f in sorted(missing_files):
        print(f"  [warn] corpus file not found, skipped every query needing it: {corpus_root / f}")

    documents = {
        file_path: (corpus_root / file_path).read_text(encoding="utf-8") for file_path in selected_docs
    }

    return documents, selected_queries

def list_first_n_documents(corpus_root, subdir, n):
    """Returns the first n document file_paths (e.g. "cuad/<name>.txt",
    relative to corpus_root -- the same form used in benchmark json and in
    chunks_by_doc / gt_spans elsewhere) from corpus_root/subdir, sorted by
    filename. Sorted-filename order is arbitrary but deterministic and
    reproducible, unlike load_benchmark's shuffle-driven selection."""
    files = sorted((corpus_root / subdir).glob("*.txt"))
    return [f"{subdir}/{f.name}" for f in files[:n]]


def load_benchmark_fixed_docs(benchmark_path, corpus_root, doc_subset, num_queries):
    """Like load_benchmark, but the document set is FIXED (doc_subset) rather
    than grown greedily to fit queries. Walks the benchmark's tests in
    their original order and keeps a query only if EVERY one of its ground
    truth snippets falls inside doc_subset -- a query that needs even one
    document outside the subset is skipped entirely, never truncated to
    its in-subset snippets, since that would silently understate its true
    ground truth. Stops once num_queries queries are collected.

    Pair with list_first_n_documents() to get "first N documents" behavior;
    doc_subset can also be any explicit list of file_paths you want to test.
    """
    raw = json.loads(benchmark_path.read_text(encoding="utf-8"))
    tests = raw["tests"] if "tests" in raw else raw
    doc_subset = set(doc_subset)

    selected_queries = []
    for test in tests:
        if len(selected_queries) >= num_queries:
            break
        snippet_docs = {s["file_path"] for s in test["snippets"]}
        if not snippet_docs <= doc_subset:
            continue  # needs a document outside the fixed subset -- skip, don't truncate
        spans = [{"file_path": s["file_path"], "start": s["span"][0], "end": s["span"][1]}
                 for s in test["snippets"]]
        selected_queries.append({"query": test["query"], "spans": spans})

    missing = {d for d in doc_subset if not (corpus_root / d).exists()}
    for f in sorted(missing):
        print(f"  [warn] corpus file not found: {corpus_root / f}")
    documents = {
        d: (corpus_root / d).read_text(encoding="utf-8")
        for d in sorted(doc_subset) if d not in missing
    }
    if not selected_queries:
        print(f"  [warn] no queries found whose ground truth lies entirely within "
              f"the {len(doc_subset)} fixed documents -- try more documents.")
    return documents, selected_queries
