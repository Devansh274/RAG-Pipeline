"""Sanity check - verify chunk offsets really point at the chunk text."""

import random
from config import FIXED_CHUNK_OVERLAP, STRUCTURE_CHUNK_OVERLAP


def spot_check_offsets(chunks_by_doc, documents, sample_size=200, seed=0, show=5):
    """Verify documents[file_path][start:end] == chunk text for a random
    sample of chunks. Returns (pct_correct, failures)."""
    all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]
    sample = random.Random(seed).sample(all_chunks, min(sample_size, len(all_chunks)))

    failures, ambiguous = [], 0
    for c in sample:
        doc = documents[c["file_path"]]
        in_bounds = 0 <= c["start"] < c["end"] <= len(doc)
        if not in_bounds or doc[c["start"]:c["end"]] != c["text"]:
            failures.append(c)
        elif doc.count(c["text"]) > 1:
            ambiguous += 1   # text recurs, so the recorded offset could be a duplicate's

    pct = 100 * (len(sample) - len(failures)) / len(sample) if sample else 0.0
    print(f"Offset spot-check: {len(sample) - len(failures)}/{len(sample)} correct ({pct:.1f}%)"
          f"  [{ambiguous} correct but with repeated text]")
    for c in failures[:show]:
        doc = documents[c["file_path"]]
        print(f"  FAIL {c['file_path']} [{c['start']}:{c['end']}]")
        print(f"    chunk : {c['text'][:80]!r}")
        print(f"    source: {doc[c['start']:c['end']][:80]!r}")
    return pct, failures

def check_chunk_overlaps(chunks_by_doc, show=5):
    """Flag chunks that overlap their predecessor by more than the configured
    overlap -- a sign that offset recovery matched an earlier duplicate."""
    max_ov = max(FIXED_CHUNK_OVERLAP, STRUCTURE_CHUNK_OVERLAP)
    bad = 0
    for fp, chunks in chunks_by_doc.items():
        for a, b in zip(chunks, chunks[1:]):
            if b["start"] < a["end"] - max_ov:
                bad += 1
                if bad <= show:
                   print(f"  SUSPECT {fp[:40]}: {(a['start'], a['end'])} -> {(b['start'], b['end'])}  "
                        f"len={len(b['text'])} text={b['text'][:30]!r}")
    print(f"Chunk-overlap check: {bad} chunk(s) overlap their predecessor "
          f"by more than {max_ov} chars")
    return bad
