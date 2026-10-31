"""Step 3 - embedding backends (Voyage, Cohere, local) with lazy clients,
retry/backoff, and a LangChain-compatible wrapper (ProviderEmbeddings).
Also owns the Anthropic/Gemini client getters, which reranking.py reuses."""

import os
import time

import numpy as np
from langchain_core.embeddings import Embeddings

# NOTE: voyageai, cohere, sentence_transformers, anthropic and google.genai are
# imported lazily INSIDE the _get_*_client functions, so you only need the
# packages for the providers you actually use.


_voyage_client = None
_cohere_client = None
_local_models = {}
_anthropic_client = None
_gemini_client = None


def _get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        import voyageai
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("Set the VOYAGE_API_KEY environment variable.")
        _voyage_client = voyageai.Client(api_key=api_key)
    return _voyage_client


def _get_cohere_client():
    global _cohere_client
    if _cohere_client is None:
        import cohere
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("Set the COHERE_API_KEY environment variable.")
        _cohere_client = cohere.ClientV2(api_key=api_key)
    return _cohere_client


def _get_local_model(name):
    if name not in _local_models:
        from sentence_transformers import SentenceTransformer
        _local_models[name] = SentenceTransformer(name, trust_remote_code=True)
    return _local_models[name]


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable.")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set the GEMINI_API_KEY environment variable.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _batched(items, size=90):
    for i in range(0, len(items), size):
        yield items[i:i + size]

def _is_retryable_error(exc):
    """True if exc looks like a transient error worth retrying: a 429 rate/
    quota limit, or the provider's servers being temporarily overloaded
    (503/500/529 -- e.g. Gemini's "high demand" ServerError, Anthropic's
    "overloaded_error"). voyageai, cohere, anthropic, and google-genai each
    raise their own exception types, so check status_code/code first and
    fall back to the exception's class name / message for SDK versions that
    only surface it that way.

    Note this does NOT distinguish a per-minute rate limit (worth retrying)
    from a per-day quota exhausted (retrying won't help until it resets) --
    both surface as 429s. If you see repeated retries ending in a 429 whose
    message mentions a daily/RPD quota, the fix is a model with a larger
    free-tier allowance or enabling billing, not waiting longer.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)  # google-genai's ClientError/ServerError use .code
    if status in (429, 500, 503, 529):
        return True
    name = type(exc).__name__.lower()
    if any(s in name for s in ("toomanyrequests", "ratelimit", "overloaded",
                                "unavailable", "serviceunavailable")):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in ("429", "rate limit", "503", "unavailable", "overloaded"))


def _call_with_backoff(fn, *args, max_retries=6, base_delay=20, **kwargs):
    """Calls fn(*args, **kwargs), retrying with exponential backoff ONLY on
    retryable errors (see _is_retryable_error) -- anything else is raised
    immediately. Needed because a provider's per-minute token limit (or a
    trial key's stricter limits) can be hit even when nothing about the
    request is wrong, and because providers' servers do occasionally return
    a transient 503 under load; the right response to either is to wait and
    retry, not to fail the whole run."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_retryable_error(exc) or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"    [rate limited] retrying in {delay}s "
                  f"(attempt {attempt + 1}/{max_retries})...")
            time.sleep(delay)


def embed_voyage(texts, model_name, input_type):
    """Embed texts with Voyage AI using document/query-specific input types."""
    client = _get_voyage_client()
    vectors = []
    for batch in _batched(texts):
        resp = _call_with_backoff(
            client.embed,
            batch,
            model=model_name,
            input_type=input_type,
            output_dtype="float",
        )
        vectors.extend(resp.embeddings)
    arr = np.array(vectors, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)  # defensive normalize
    return arr


def embed_cohere(texts, model_name, input_type):
    client = _get_cohere_client()
    vectors = []
    for batch in _batched(texts):
        resp = _call_with_backoff(
            client.embed,
            texts=batch,
            model=model_name,
            input_type=input_type,
            embedding_types=["float"],
        )
        vectors.extend(resp.embeddings.float)
    arr = np.array(vectors, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def embed_local(texts, model_name):
    model = _get_local_model(model_name)
    arr = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(arr, dtype=np.float32)


def _embed_chunks_raw(provider, model_name, chunks):
    """Embeds chunk text with the given provider/model. Returns an
    (n_chunks, dim) array aligned with `chunks`. This is the same
    per-provider dispatch as before (renamed with a leading underscore) —
    ProviderEmbeddings below is a thin interface wrapper around it, not a
    replacement for it."""
    texts = [c["text"] for c in chunks]
    if provider == "voyage":
        return embed_voyage(texts, model_name, input_type="document")
    if provider == "cohere":
        return embed_cohere(texts, model_name, input_type="search_document")
    if provider == "local":
        return embed_local(texts, model_name)
    raise ValueError(f"Unknown provider: {provider}")


def _embed_query_raw(provider, model_name, query):
    if provider == "voyage":
        return embed_voyage([query], model_name, input_type="query")[0]
    if provider == "cohere":
        return embed_cohere([query], model_name, input_type="search_query")[0]
    if provider == "local":
        return embed_local([query], model_name)[0]
    raise ValueError(f"Unknown provider: {provider}")


def _embed_queries_raw(provider, model_name, queries):
    """Embeds every query in `queries` in one pass, batched the same way
    _embed_chunks_raw batches documents (via embed_voyage/embed_cohere's own
    _batched loop). Used to embed all selected queries up front instead of
    calling _embed_query_raw once per query inside the per-query eval loop
    -- that per-query pattern means one network call per query (80 calls
    for NUM_QUERIES=80), which burns through a provider's per-minute token
    budget and, for a trial API key, its total call quota, much faster than
    a handful of batched calls covering the same queries."""
    if provider == "voyage":
        return embed_voyage(queries, model_name, input_type="query")
    if provider == "cohere":
        return embed_cohere(queries, model_name, input_type="search_query")
    if provider == "local":
        return embed_local(queries, model_name)
    raise ValueError(f"Unknown provider: {provider}")


class ProviderEmbeddings(Embeddings):
    """LangChain `Embeddings` adapter around this script's existing
    embed_voyage / embed_cohere / embed_local functions.

    This is intentionally a thin pass-through, not a reimplementation: the
    provider-specific API calls, batching (_batched), input_type selection
    (document vs query, and each provider's own naming for that), and
    normalization all stay exactly as they were in _embed_chunks_raw /
    _embed_query_raw. Wrapping them here only gives the embedding step a
    standard interface — useful if you plug in a LangChain VectorStore later
    (deliverable 2's /search endpoint) — without changing which model does
    the embedding or how its output is computed.
    """

    def __init__(self, provider, model_name):
        self.provider = provider
        self.model_name = model_name

    def embed_documents(self, texts):
        pseudo_chunks = [{"text": t} for t in texts]
        return _embed_chunks_raw(self.provider, self.model_name, pseudo_chunks).tolist()

    def embed_query(self, text):
        return _embed_query_raw(self.provider, self.model_name, text).tolist()
