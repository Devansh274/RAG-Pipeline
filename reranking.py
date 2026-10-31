"""Step 5 (optional) - LLM reranking of the cosine-similarity candidates."""

import json
import re

from config import RERANK_PROVIDER, RERANK_MODEL, RERANK_MAX_CHARS_PER_CHUNK
from embeddings import _call_with_backoff, _get_anthropic_client, _get_gemini_client

# NOTE: `from google.genai import types` is imported lazily inside
# _rerank_call_gemini, so the google-genai package is only needed for Gemini.


# by cosine similarity, ask an LLM to judge each chunk's relevance to the
# query directly (reading the actual text, not comparing embedding vectors)
# and re-sort by that judgment. This is a classic "cross-encoder"-style
# reranking step -- the LLM sees the query and the passage together in one
# prompt, which lets it catch relevance the embedding model missed (e.g. a
# chunk that's topically similar but doesn't answer the specific question
# asked, or the reverse: a chunk that's lexically distant from the query but
# is clearly the right answer once you actually read it).
#
# This is done LISTWISE, one LLM call per query covering the whole
# candidate pool, rather than POINTWISE (a separate call per chunk asking
# "how relevant is this one"). Pointwise is simpler to reason about and
# slightly more robust to any one bad score, but for NUM_QUERIES=80 and a
# ~20-chunk pool it's 1,600 LLM calls instead of 80 -- roughly 20x the cost
# and latency for a benchmark run. Listwise scoring also lets the model
# weigh candidates against each other, not just against the query in
# isolation, which tends to help it calibrate scores consistently.

def _parse_rerank_scores(raw_text, n_chunks):
    """Parse the LLM's response into a list of n_chunks scores, indexed to
    match the order chunks were shown in the prompt. Tolerant of a ```json
    fence around the array (some models add one despite being told not to).

    Any index the model skipped, duplicated, or returned out of range keeps
    a fallback score instead of raising: real scores win, and any unscored
    chunk is placed just below the lowest real score, in its original
    cosine-rank order relative to other unscored chunks. That way a partial
    or slightly malformed response still produces a usable ranking (real
    judgments first, un-judged chunks falling back to cosine order) instead
    of losing the whole query to a formatting glitch.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)

    scores = [None] * n_chunks
    try:
        parsed = json.loads(cleaned)
        for item in parsed:
            idx = int(item["index"])
            if 0 <= idx < n_chunks and scores[idx] is None:
                scores[idx] = float(item["score"])
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        pass  # everything stays None -> all chunks get the fallback below

    real_scores = [s for s in scores if s is not None]
    floor = (min(real_scores) - 1) if real_scores else 0.0
    for i in range(n_chunks):
        if scores[i] is None:
            scores[i] = floor - (i * 1e-6)  # preserves original relative order
    return scores


def _build_rerank_prompt(query, chunks, max_chars):
    passages_block = "\n\n".join(
        f"[{i}] {c['text'][:max_chars]}" for i, c in enumerate(chunks)
    )
    return (
        "You are scoring passages from legal documents for relevance to a search "
        "query, to rerank retrieval results for a legal RAG system.\n\n"
        "Score each passage from 0 to 10:\n"
        "- 0-2: irrelevant, or only shares surface vocabulary with the query\n"
        "- 3-5: topically related but does not answer what the query is asking\n"
        "- 6-8: partially answers the query, or contains relevant but incomplete info\n"
        "- 9-10: directly contains the specific clause/fact/answer the query needs\n\n"
        f"Query: {query}\n\n"
        f"Passages:\n{passages_block}\n\n"
        "Respond with ONLY a JSON array, no other text before or after it, of "
        'objects {"index": <passage number>, "score": <0-10>}, one entry per '
        "passage, covering every passage index exactly once."
    )


def _rerank_call_anthropic(prompt, model_name):
    """Returns the model's raw text response for the rerank prompt."""
    client = _get_anthropic_client()
    resp = _call_with_backoff(
        client.messages.create,
        model=model_name,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _rerank_call_gemini(prompt, model_name):
    """Returns the model's raw text response for the rerank prompt.

    Sets response_mime_type="application/json" (Gemini-specific constrained
    decoding), which is stricter than just asking nicely in the prompt text
    and makes the JSON-parsing fallback in _parse_rerank_scores a rarely-hit
    safety net rather than the norm.
    """
    from google.genai import types
    client = _get_gemini_client()
    resp = _call_with_backoff(
        client.models.generate_content,
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    return resp.text or ""


_RERANK_CALL_FNS = {
    "anthropic": _rerank_call_anthropic,
    "gemini": _rerank_call_gemini,
}


def llm_rerank(query, chunks, provider=RERANK_PROVIDER, model_name=RERANK_MODEL,
                max_chars=RERANK_MAX_CHARS_PER_CHUNK):
    """chunks: a candidate pool (list of chunk dicts), already in cosine-
    similarity order. Returns (reranked_chunks, scores) -- the same chunks,
    reordered by descending LLM-judged relevance, and the score assigned to
    each (aligned to the returned order, not the input order).

    `provider` selects which underlying API does the scoring; the prompt,
    parsing, and everything else here are shared across providers -- only
    the raw API call (_rerank_call_anthropic / _rerank_call_gemini) differs.
    """
    if not chunks:
        return [], []
    if provider not in _RERANK_CALL_FNS:
        raise ValueError(f"Unknown RERANK_PROVIDER: {provider!r}")

    prompt = _build_rerank_prompt(query, chunks, max_chars)
    raw_text = _RERANK_CALL_FNS[provider](prompt, model_name)
    scores = _parse_rerank_scores(raw_text, len(chunks))

    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in order], [scores[i] for i in order]
