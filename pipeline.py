"""The orchestrator: run() wires every step together
(load -> chunk -> embed/index -> retrieve -> rerank -> evaluate -> summarize)."""

from config import (
    ANSWER_PREVIEW_CHARS,
    BENCHMARK_PATH,
    CHUNKING_SCHEME,
    CORPUS_ROOT,
    FIXED_DOCUMENT_SUBDIR,
    MAX_DOCUMENTS,
    METADATA_METHOD,
    METADATA_QUERY_SEPARATOR,
    METADATA_RETRIEVAL_ENABLED,
    METADATA_TOP_DOCS,
    MODELS,
    NUM_QUERIES,
    RERANK_CANDIDATE_POOL,
    RERANK_ENABLED,
    TOP_K,
    USE_CHROMA,
    USE_FIXED_DOCUMENT_SUBSET,
)
from data_loader import list_first_n_documents, load_benchmark, load_benchmark_fixed_docs
from chunking import _structure_boundaries, chunk_documents, fixed_chunk, structure_chunk
from sanity_checks import spot_check_offsets
from embeddings import ProviderEmbeddings, _embed_queries_raw
from sanity_checks import spot_check_offsets, check_chunk_overlaps
from retrieval import (
    DocumentMetadataIndex,
    _doc_to_chunk,
    build_chroma_collection,
    build_vector_store,
    chroma_search,
    split_query,
)
from reranking import llm_rerank
from evaluation import (
    _merged_intervals_by_file,
    chunk_is_hit,
    evaluate_query,
    intersect_intervals,
    interval_set_len,
    merge_intervals,
    micro_prf,
)


def run():
    if USE_FIXED_DOCUMENT_SUBSET:
        doc_subset = list_first_n_documents(CORPUS_ROOT, FIXED_DOCUMENT_SUBDIR, MAX_DOCUMENTS)
        documents, selected_queries = load_benchmark_fixed_docs(
        BENCHMARK_PATH, CORPUS_ROOT, doc_subset, NUM_QUERIES
    )
    else:
        documents, selected_queries = load_benchmark(
        BENCHMARK_PATH, CORPUS_ROOT, MAX_DOCUMENTS, NUM_QUERIES
    )
    print(f"Loaded {len(documents)} document(s), {len(selected_queries)} quer{'y' if len(selected_queries) == 1 else 'ies'}.")
    for fp in documents:
        print(f"  - {fp}")

    chunk_fn = {"fixed": fixed_chunk, "structure": structure_chunk}[CHUNKING_SCHEME]
    chunks_by_doc = chunk_documents(documents, chunk_fn)
    pct, _ = spot_check_offsets(chunks_by_doc, documents)
    assert pct >= 90, f"Offset spot-check below the 90% success criterion: {pct:.1f}%"
    check_chunk_overlaps(chunks_by_doc)
    for fp, chunks in chunks_by_doc.items():
        print(f"  {fp}: {len(chunks)} chunks ({CHUNKING_SCHEME} chunking)")
        if CHUNKING_SCHEME == "structure":
            struct_boundaries, method = _structure_boundaries(documents[fp])
            n_boundaries = max(0, len(struct_boundaries) - 2)
            print(f"    {n_boundaries} structural boundaries detected (method: {method})")
            if method == "none (fixed-size fallback)":
                print("    [warn] No headings, blank-line paragraphs, line breaks, or sentence "
                      "punctuation patterns were found in this document at all — it's being "
                      "split with the fixed-size fallback inside structure_chunk. Inspect the "
                      "raw extracted text for this file; this is the one case where 'structure' "
                      "chunking is genuinely no better than 'fixed'.")

    # Stage-1 setup (once per run, shared by all models): split every query
    # into (hint, question) and index the documents' file name + header.
    all_query_texts = [q["query"] for q in selected_queries]
    parsed_queries = [split_query(t) for t in all_query_texts]
    meta_index = DocumentMetadataIndex(documents) if METADATA_RETRIEVAL_ENABLED else None
    if METADATA_RETRIEVAL_ENABLED:
        n_with_hint = sum(1 for h, _ in parsed_queries if h)
        print(f"\nTwo-stage retrieval ON (metadata method: {METADATA_METHOD}, "
              f"top {METADATA_TOP_DOCS} docs). {n_with_hint}/{len(parsed_queries)} queries "
              f"have a '{METADATA_QUERY_SEPARATOR}' hint; the rest search all documents.")
    else:
        print("\nTwo-stage retrieval OFF (full query, whole corpus).")

    summary = {}

    for model_cfg in MODELS:
        provider, model_name = model_cfg["provider"], model_cfg["name"]
        label = f"{provider}/{model_name}"
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")

        try:
            if USE_CHROMA:
                vector_store = build_chroma_collection(provider, model_name, chunks_by_doc)
            else:
                embeddings_model = ProviderEmbeddings(provider, model_name)
                vector_store = build_vector_store(embeddings_model, chunks_by_doc)

            all_metrics = {"char_precision@k": [], "char_recall@k": [], "char_f1@k": [],
                           "hit_rate": [], "top1_hit_rate": [], "mrr": []}   # added top1_hit_rate
            all_metrics_rerank = {m: [] for m in all_metrics} if RERANK_ENABLED else None
            rerank_failures = 0
            char_totals = {"overlap_chars": 0, "retrieved_chars": 0, "gt_chars": 0}
            char_totals_rerank = dict.fromkeys(char_totals, 0)
            doc_recalls, subset_sizes = [], []   # stage-1 diagnostics

            # Embed every selected query once, up front, in a handful of
            # batched calls (see _embed_queries_raw) rather than one API
            # call per query inside the loop below.
            #
            # query_texts are the full original queries (used for display and
            # the LLM reranker). The vectors used for chunk similarity come
            # from the QUESTION part only when two-stage retrieval is on.
            query_texts = all_query_texts
            sim_texts = ([q for _, q in parsed_queries]
                         if METADATA_RETRIEVAL_ENABLED else query_texts)
            n_batches = (len(sim_texts) + 89) // 90
            print(f"  Embedding {len(sim_texts)} quer{'y' if len(sim_texts) == 1 else 'ies'} "
                  f"in {n_batches} batch call(s)...")
            query_vectors = _embed_queries_raw(provider, model_name, sim_texts)

            # Stage 1: choose the candidate document subset for every query.
            if METADATA_RETRIEVAL_ENABLED:
                hint_vecs = {}
                if METADATA_METHOD in ("embedding", "hybrid"):
                    meta_index.embed(provider, model_name)
                    hint_idx = [i for i, (h, _) in enumerate(parsed_queries) if h]
                    if hint_idx:
                        vecs = _embed_queries_raw(
                            provider, model_name, [parsed_queries[i][0] for i in hint_idx])
                        hint_vecs = dict(zip(hint_idx, vecs))
                subsets = [meta_index.select(h, hint_vecs.get(i))
                           for i, (h, _) in enumerate(parsed_queries)]
            else:
                subsets = [(list(documents), None)] * len(query_texts)

            for qi, q in enumerate(selected_queries):
                gt_spans = q["spans"]
                # Stage 1 result for this query: the candidate document subset.
                subset_docs, subset_note = subsets[qi]
                subset_set = set(subset_docs)
                gt_docs = {s["file_path"] for s in gt_spans}
                missed_docs = gt_docs - subset_set
                doc_recalls.append(1 - len(missed_docs) / len(gt_docs))
                subset_sizes.append(len(subset_docs))
                total_candidates = sum(len(chunks_by_doc[fp]) for fp in subset_docs)

                # Stage 2: plain top-k similarity search via the vector store,
                # scoped to the stage-1 subset, using the query vector embedded
                # above (the question part, when two-stage is on). Asking for
                # all candidates in the subset (not just TOP_K) gives us the
                # full ranking for the diagnostics below at no extra embedding
                # cost -- it's the same query vector, just a bigger k.
                subset_filter = (None if len(subset_docs) == len(documents) else
                                 (lambda doc, _refs=subset_set: doc.metadata["file_path"] in _refs))
                if USE_CHROMA:
                    ranked = chroma_search(
                    vector_store, query_vectors[qi], total_candidates,
                    None if len(subset_docs) == len(documents) else subset_docs)
                else:
                    ranked = vector_store.similarity_search_with_score_by_vector(
                    query_vectors[qi].tolist(),
                    k=total_candidates,
                    filter=subset_filter,
                )
                full_order_chunks = [_doc_to_chunk(doc) for doc, _ in ranked]
                full_order_scores = [score for _, score in ranked]
                retrieved = full_order_chunks[:TOP_K]

                # Diagnostic: rank of the first chunk (in the FULL ranking, not
                # just top-k) that overlaps the ground truth at all. Distinguishes
                # "the answer chunk exists but ranked below top-k" (a retrieval/
                # ranking issue) from "no chunk in the pool overlaps the ground
                # truth at all" (a chunking issue — the answer got cut apart or
                # was never captured as a coherent chunk to begin with).
                gt_by_file_full = _merged_intervals_by_file(gt_spans)
                first_hit_rank = None
                for rank, c in enumerate(full_order_chunks, start=1):
                    if chunk_is_hit(c, gt_by_file_full):
                        first_hit_rank = rank
                        break

                metrics = evaluate_query(retrieved, gt_spans, k=TOP_K)
                for m in all_metrics:
                    all_metrics[m].append(metrics[m])
                for key in char_totals:
                    char_totals[key] += metrics[key]

                # Second stage: hand the top RERANK_CANDIDATE_POOL cosine hits to
                # the LLM, re-sort by its relevance judgment, take the new top-k,
                # and score that the same way. If the LLM call itself fails (bad
                # key, rate limit exhausted, network) we fall back to the cosine
                # metrics for this one query rather than crashing the whole run
                # or silently dropping the query from the reranked average.
                reranked_retrieved, metrics_rerank = None, None
                if RERANK_ENABLED:
                    candidate_pool = full_order_chunks[:RERANK_CANDIDATE_POOL]
                    try:
                        reranked_chunks, _rerank_scores = llm_rerank(query_texts[qi], candidate_pool)
                        reranked_retrieved = reranked_chunks[:TOP_K]
                        metrics_rerank = evaluate_query(reranked_retrieved, gt_spans, k=TOP_K)
                    except Exception as e:
                        rerank_failures += 1
                        print(f"    [rerank failed] {type(e).__name__}: {e} "
                              f"-- using cosine-only metrics for this query's rerank average")
                        reranked_retrieved, metrics_rerank = retrieved, metrics
                    for m in all_metrics_rerank:
                        all_metrics_rerank[m].append(metrics_rerank[m])
                    for key in char_totals_rerank:
                        char_totals_rerank[key] += metrics_rerank[key]

                gt_by_file = _merged_intervals_by_file(gt_spans)

                print(f"\n  Query: {query_texts[qi]}")
                if METADATA_RETRIEVAL_ENABLED:
                    hint_q, question_q = parsed_queries[qi]
                    shown = (subset_docs if len(subset_docs) <= max(METADATA_TOP_DOCS, 1)
                             else f"all {len(subset_docs)} documents")
                    print(f"  Stage 1 (metadata/{METADATA_METHOD}) hint: {hint_q!r} -> "
                          f"{len(subset_docs)}/{len(documents)} doc(s): {shown}")
                    if subset_note:
                        print(f"    [{subset_note}]")
                    if missed_docs:
                        print(f"    [metadata miss] ground-truth doc(s) not in subset: "
                              f"{sorted(missed_docs)}")
                    print(f"  Stage 2 (similarity) question: {question_q!r}")
                print(f"  Ground truth: {[(s['file_path'], s['start'], s['end']) for s in gt_spans]}"
                f" ({metrics['gt_chars']} chars total)")
                # for s in gt_spans:
                #  print(f"    >>> {documents[s['file_path']][s['start']:s['end']]}")
                if first_hit_rank is None and missed_docs:
                    print(f"  [diagnostic] stage 1 dropped the ground-truth document, so none of "
                          f"the {len(full_order_chunks)} chunks searched could contain the answer "
                          f"(a metadata-retrieval miss, not a chunking/ranking one).")
                elif first_hit_rank is None:
                    print(f"  [diagnostic] NO chunk among all {len(full_order_chunks)} candidate "
                          f"chunks overlaps the ground truth at all — the answer text isn't "
                          f"captured by any generated chunk.")
                elif first_hit_rank > TOP_K:
                    print(f"  [diagnostic] first semantic hit ranked #{first_hit_rank} of "
                          f"{len(full_order_chunks)}; plain top_k={TOP_K} retrieval missed it.")
                for rank, c in enumerate(retrieved, start=1):
                    c_gt = gt_by_file.get(c["file_path"], [])
                    c_overlap = interval_set_len(intersect_intervals(
                        merge_intervals([(c["start"], c["end"])]), c_gt))
                    preview = c["text"][:ANSWER_PREVIEW_CHARS].replace("\n", " ").strip()
                    print(f"    [{rank}] {c_overlap:4d} chars overlap  "
                          f"{c['file_path']} [{c['start']}:{c['end']}] score={full_order_scores[rank - 1]:.3f}")
                    print(f"         {preview}...")

                if RERANK_ENABLED and reranked_retrieved is not None and reranked_retrieved is not retrieved:
                    print(f"    -- after LLM rerank (pool of {len(candidate_pool)} cosine candidates) --")
                    for rank, c in enumerate(reranked_retrieved, start=1):
                        c_gt = gt_by_file.get(c["file_path"], [])
                        c_overlap = interval_set_len(intersect_intervals(
                            merge_intervals([(c["start"], c["end"])]), c_gt))
                        preview = c["text"][:ANSWER_PREVIEW_CHARS].replace("\n", " ").strip()
                        print(f"    [{rank}] {c_overlap:4d} chars overlap  "
                              f"{c['file_path']} [{c['start']}:{c['end']}]")
                        print(f"         {preview}...")

                if RERANK_ENABLED:
                    print(f"  Metrics (cosine only):   {metrics}")
                    print(f"  Metrics (+ LLM rerank):  {metrics_rerank}")
                else:
                    print(f"  Metrics: {metrics}")

            avg = {m: round(sum(v) / len(v), 3) for m, v in all_metrics.items()}
            avg.update(micro_prf(**char_totals)) 
            doc_stats = {
                "doc_recall": round(sum(doc_recalls) / len(doc_recalls), 3),
                "avg_docs": round(sum(subset_sizes) / len(subset_sizes), 1),
            }
            print(f"\n  --- {label} (cosine only) averages over {len(all_metrics['hit_rate'])} queries: {avg} ---")
            print(f"  --- stage-1 document recall: {doc_stats['doc_recall']} "
                  f"(avg {doc_stats['avg_docs']} of {len(documents)} docs searched per query) ---")
            summary[label] = {**avg, **doc_stats}

            if RERANK_ENABLED:
                avg_rerank = {m: round(sum(v) / len(v), 3) for m, v in all_metrics_rerank.items()}
                avg_rerank.update(micro_prf(**char_totals_rerank))   
                rerank_label = f"{label} + LLM rerank"
                print(f"  --- {rerank_label} averages over {len(all_metrics_rerank['hit_rate'])} queries: {avg_rerank} ---")
                if rerank_failures:
                    print(f"  [note] {rerank_failures} of {len(selected_queries)} quer"
                          f"{'y' if rerank_failures == 1 else 'ies'} fell back to cosine-only "
                          f"metrics above (the LLM rerank call failed).")
                summary[rerank_label] = {**avg_rerank, **doc_stats}

        except Exception as e:
            # Don't let one provider's failure (bad/expired key, no credits,
            # rate limit, network issue, etc.) take down the whole comparison
            # — report it and move on to the next model.
            print(f"\n  [SKIPPED] {label} failed: {type(e).__name__}: {e}")
            continue
        
    ws = sum(1 for cs in chunks_by_doc.values() for c in cs if not c["text"].strip())
    print(f"Whitespace-only chunks: {ws}")
    print(f"\n{'=' * 60}\nSUMMARY (avg over all {len(selected_queries)} queries)\n{'=' * 60}")
    print(f"{'model':56s} {'char_prec@k':>11s} {'char_rec@k':>11s} {'char_f1@k':>10s} "
          f"{'hit_rate':>9s} {'mrr':>6s} {'top1_hit':>9s} {'doc_rec':>8s} {'avg_docs':>9s}"
          f"{'mic_prec':>9s} {'mic_rec':>8s} {'mic_f1':>7s}")
    for label, avg in summary.items():
        print(f"{label:56s} {avg['char_precision@k']:11.3f} {avg['char_recall@k']:11.3f} "
              f"{avg['char_f1@k']:10.3f} {avg['hit_rate']:9.3f} {avg['mrr']:6.3f} {avg['top1_hit_rate']:9.3f} "
              f"{avg['doc_recall']:8.3f} {avg['avg_docs']:9.1f}"
              f"{avg['micro_precision@k']:9.3f} {avg['micro_recall@k']:8.3f} "
              f"{avg['micro_f1@k']:7.3f}")
