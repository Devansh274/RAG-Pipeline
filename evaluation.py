"""Step 6 - character-level precision/recall/F1, hit-rate and MRR scoring
against LegalBench-RAG's character-offset ground truth."""

from config import TOP_K, MIN_HIT_OVERLAP_FRAC


# retrieved chunks and ground-truth snippets as sets of character positions
# (per document) and score the actual overlap in characters, not just
# whether a chunk happened to touch a span at all. This matters because a
# chunk that overlaps 3 characters of a 400-character ground-truth span and
# a chunk that overlaps all 400 are very different retrievals, and chunk-
# level hit/miss scoring can't tell them apart.

def merge_intervals(intervals):
    """intervals: list of (start, end) tuples, same document. Returns them
    sorted and merged so overlapping/adjacent ranges don't get double-counted."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def interval_set_len(intervals):
    """Total character count covered by a list of (already merged, but works
    either way since it just sums) non-overlapping intervals."""
    return sum(e - s for s, e in intervals)


def intersect_intervals(a, b):
    """a, b: merged, sorted (start, end) interval lists. Returns their
    overlapping regions as a new interval list."""
    i, j, out = 0, 0, []
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _merged_intervals_by_file(items):
    """items: list of {"file_path","start","end"} (chunks or gt spans).
    Returns {file_path: merged interval list}."""
    by_file = {}
    for it in items:
        by_file.setdefault(it["file_path"], []).append((it["start"], it["end"]))
    return {fp: merge_intervals(ivs) for fp, ivs in by_file.items()}

def chunk_is_hit(chunk, gt_by_file, min_frac=MIN_HIT_OVERLAP_FRAC):
    gt_ivs = gt_by_file.get(chunk["file_path"], [])
    overlap = interval_set_len(
        intersect_intervals([(chunk["start"], chunk["end"])], gt_ivs))
    if overlap <= 0:
        return False
    best_possible = min(chunk["end"] - chunk["start"], interval_set_len(gt_ivs))
    return overlap >= min_frac * best_possible

def evaluate_query(retrieved_chunks, ground_truth_spans, k=TOP_K):
    """retrieved_chunks: top-k chunks for one query, already ranked.
    ground_truth_spans: list of {"file_path","start","end"} — every snippet
    the benchmark marks as relevant for this query.

    Precision/recall/F1 are computed on character overlap (unioned across
    chunks/spans within a document, so overlapping chunks don't inflate the
    count). hit_rate/mrr stay rank-based (did/when did we retrieve *any*
    overlapping character) since those are inherently about ranking, not
    overlap magnitude.
    """
    top_k = retrieved_chunks[:k]

    retrieved_by_file = _merged_intervals_by_file(top_k)
    gt_by_file = _merged_intervals_by_file(ground_truth_spans)

    retrieved_chars = sum(interval_set_len(ivs) for ivs in retrieved_by_file.values())
    gt_chars = sum(interval_set_len(ivs) for ivs in gt_by_file.values())

    overlap_chars = sum(
        interval_set_len(intersect_intervals(retrieved_by_file.get(fp, []), gt_ivs))
        for fp, gt_ivs in gt_by_file.items()
    )

    char_precision = overlap_chars / retrieved_chars if retrieved_chars else 0.0
    char_recall = overlap_chars / gt_chars if gt_chars else 0.0
    char_f1 = (2 * char_precision * char_recall / (char_precision + char_recall)
               if (char_precision + char_recall) else 0.0)

    # Rank-based metrics: a chunk "hits" if it has ANY character overlap
    # with the ground truth in its own document.
    hits = [chunk_is_hit(c, gt_by_file) for c in top_k]
    hit_rate = 1.0 if any(hits) else 0.0
    mrr = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    top1_hit_rate = 1.0 if hits and hits[0] else 0.0   # NEW: rank-1 chunk overlaps ground truth

    return {
        "char_precision@k": round(char_precision, 4),
        "char_recall@k": round(char_recall, 4),
        "char_f1@k": round(char_f1, 4),
        "hit_rate": hit_rate,
        "top1_hit_rate": top1_hit_rate, 
        "mrr": mrr,
        "overlap_chars": overlap_chars,
        "retrieved_chars": retrieved_chars,
        "gt_chars": gt_chars,
    }

def micro_prf(overlap_chars, retrieved_chars, gt_chars):
    """Micro-averaged P/R/F1 from character counts summed over all queries."""
    p = overlap_chars / retrieved_chars if retrieved_chars else 0.0
    r = overlap_chars / gt_chars if gt_chars else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"micro_precision@k": round(p, 3),
            "micro_recall@k": round(r, 3),
            "micro_f1@k": round(f1, 3)}
