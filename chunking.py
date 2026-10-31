"""Step 2 - split documents into chunks (fixed-size or structure-aware).
Every chunk is a dict: {"text", "start", "end"} with exact character offsets."""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    FIXED_CHUNK_SIZE,
    FIXED_CHUNK_OVERLAP,
    STRUCTURE_MAX_CHUNK_SIZE,
    STRUCTURE_CHUNK_OVERLAP,
)


def _split_with_offsets(splitter, text):
    """Run a LangChain text splitter and recover exact character offsets for
    each returned chunk by locating it as a literal substring of `text`,
    searching forward from just past the previous chunk's start.

    We recover offsets this way instead of trusting the splitter's built-in
    `add_start_index` metadata: that feature has a documented bug where it
    returns the wrong start_index whenever a chunk's text recurs elsewhere
    earlier in the document (common in legal boilerplate — repeated defined
    terms, repeated clause headers). Since LegalBench-RAG scoring is entirely
    character-offset-based, a silently wrong offset there would corrupt every
    downstream metric exactly the way the old fixed_chunk bug did. Searching
    forward from the previous match, rather than from position 0 each time,
    sidesteps that failure mode: chunk starts are always monotonically
    increasing, so we never re-match an earlier, wrong occurrence.

    Requires the splitter to be built with strip_whitespace=False so that
    each returned piece is an exact, unmodified substring of `text` and the
    `.find()` lookup below can always succeed.
    """
    pieces = splitter.split_text(text)
    chunks, search_from = [], 0
    for piece in pieces:
        if not piece.strip():      # whitespace-only fragment: no content, skip it
            continue
        idx = text.find(piece, search_from)
        if idx == -1:
            # Fail loudly rather than silently record a wrong offset.
            raise ValueError(
                "Could not locate a chunk's text back in its source document "
                f"during offset recovery (chunk starts: {piece[:50]!r}). This "
                "should not happen with strip_whitespace=False — check the "
                "splitter configuration."
            )
        start, end = idx, idx + len(piece)
        chunks.append({"text": piece, "start": start, "end": end})
        search_from = start + 1  # forward progress; overlap still allowed
    return chunks


def fixed_chunk(text, chunk_size=FIXED_CHUNK_SIZE, overlap=FIXED_CHUNK_OVERLAP):
    """Split `text` into overlapping windows using LangChain's
    RecursiveCharacterTextSplitter, with exact character offsets recovered
    via _split_with_offsets.

    The old hand-rolled version computed each window's start with pure
    arithmetic (`end - overlap`), which had no relationship to word or
    sentence boundaries and produced chunks starting mid-word. This version
    instead splits by trying progressively finer real separators — paragraph
    break, line break, sentence-ending punctuation, clause comma, then word
    boundary — and only ever falls back to plain character-count splitting
    if a single "word" itself exceeds chunk_size (e.g. a long URL). That
    structurally prevents mid-word/mid-sentence splits rather than papering
    over them after the fact.

    chunk_size / overlap remain character counts, not tokens, matching the
    character-offset ground truth this script evaluates against."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        strip_whitespace=False,
    )
    return _split_with_offsets(splitter, text)


def _looks_like_heading(line):
    """Heuristically identify standalone policy/article headings.

    The LegalBench privacy-policy corpus is not guaranteed to use one heading
    convention. We therefore recognize several common forms instead of relying
    on one regex such as ``1. DEFINITIONS``.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False

    # Remove common heading decoration without destroying the original text.
    cleaned = re.sub(r'^[#>*\-\u2022\s]+', '', stripped)
    cleaned = re.sub(r'[:;,.]+$', '', cleaned).strip()
    if not cleaned:
        return False

    # Numbered headings: 1. Definitions / 1) Definitions / 1.2 Security
    if re.match(r'^\d+(?:[.)]|(?:\.\d+)*\.)\s+\S', cleaned):
        return True

    # Roman/letter section labels: I. Scope / A. Information We Collect
    if re.match(r'^(?:[IVXLCDM]+|[A-Z])[.)]\s+\S', cleaned):
        return True

    words = cleaned.split()
    alpha = re.sub(r'[^A-Za-z]', '', cleaned)
    if not alpha:
        return False

    # Explicit heading-like phrases common in privacy policies.
    heading_terms = {
        # 'privacy', 'definitions', 'introduction', 'information', 'collection',
        # 'use', 'disclosure', 'sharing', 'security', 'cookies', 'retention',
        # 'rights', 'choices', 'access', 'children', 'third-party', 'third party',
        # 'contact', 'changes', 'updates', 'compliance', 'consent', 'marketing',
        # 'data', 'personal information', 'your information',
    }
    lower = cleaned.lower()
    if len(words) <= 8 and cleaned.count(',') == 0 and any(
        lower == term or lower.startswith(term + ' ') for term in heading_terms
    ):
        return True

    # ALL-CAPS headings (allow digits/punctuation, require at least one letter).
    letters = [ch for ch in cleaned if ch.isalpha()]
    if letters and sum(ch.isupper() for ch in letters) / len(letters) >= 0.85:
        return len(words) <= 12

    # Short title-like standalone lines: mostly alphabetic, title-cased, and
    # not sentence-like. This catches headings such as "How We Use Information".
    if 1 <= len(words) <= 10 and cleaned[-1] not in '.?!,' and cleaned.count(',') == 0:
        title_like = sum(1 for w in words if w[:1].isupper()) / len(words)
        if title_like >= 0.6:
            return True

    return False


def _structure_boundaries(text):
    """Return (boundaries, method) using the strongest structural signal that
    actually fires, trying progressively weaker signals until one works:

        1. headings (numbered/roman/lettered, ALL-CAPS, title-case lines)
        2. blank-line-separated paragraphs
        3. single-newline lines (docs extracted as one clause/item per line,
           with no blank line between them — common after HTML->text
           conversion of policy pages, which is what tripped this up)
        4. sentence boundaries (tolerating a trailing quote/paren/bracket
           between the period and the next sentence, e.g. `...below." We...`)
        5. genuinely nothing detected: the whole document as one segment

    Only tier 5 should ever fall through to fixed-size splitting inside
    structure_chunk, and the method label lets callers report honestly when
    that's what happened instead of implying paragraph/sentence structure
    was used when it silently wasn't.
    """
    boundaries = []
    offset = 0
    for line in text.splitlines(keepends=True):
        line_no_nl = line.rstrip('\r\n')
        if offset > 0 and _looks_like_heading(line_no_nl):
            boundaries.append(offset)
        offset += len(line)

    if boundaries:
        return sorted(set([0, *boundaries, len(text)])), "headings"

    # Tier 2: blank-line paragraphs.
    paragraph_boundaries = [0] + [m.end() for m in re.finditer(r'\n\s*\n+', text)] + [len(text)]
    paragraph_boundaries = sorted(set(paragraph_boundaries))
    if len(paragraph_boundaries) > 2:
        return paragraph_boundaries, "blank-line paragraphs"

    # Tier 3: single-newline lines. Only trust this as real structure (rather
    # than soft-wrapped prose) when most non-blank lines look like complete
    # units — i.e. end in terminal/list punctuation — and lines aren't tiny.
    lines = text.splitlines(keepends=True)
    non_blank = [l.strip() for l in lines if l.strip()]
    if non_blank:
        end_punct_frac = sum(1 for l in non_blank if l[-1] in '.!?:;') / len(non_blank)
        avg_len = sum(len(l) for l in non_blank) / len(non_blank)
        if end_punct_frac >= 0.6 and avg_len >= 30:
            line_boundaries, off = [0], 0
            for line in lines:
                off += len(line)
                if line.strip() and off < len(text):
                    line_boundaries.append(off)
            line_boundaries = sorted(set([*line_boundaries, len(text)]))
            if len(line_boundaries) > 2:
                return line_boundaries, "line breaks"

    # Tier 4: sentence boundaries.
    sentence_boundaries = [0]
    for m in re.finditer(r'(?<=[.!?])["\'\u201d\u2019\)\]]*(?:\s+|\n)(?=["\'\u201c\u2018\(\[]?[A-Z\d])', text):
        sentence_boundaries.append(m.end())
    sentence_boundaries = sorted(set([*sentence_boundaries, len(text)]))
    if len(sentence_boundaries) > 2:
        return sentence_boundaries, "sentence boundaries"

    # Tier 5: nothing structural found anywhere — no headings, no blank
    # lines, no line structure, no detectable sentence punctuation pattern.
    # This is the true last resort; structure_chunk will fixed-chunk this
    # single segment, and callers should say so rather than claiming
    # paragraph/sentence structure was used.
    return [0, len(text)], "none (fixed-size fallback)"


def structure_chunk(text, max_size=STRUCTURE_MAX_CHUNK_SIZE, overlap=STRUCTURE_CHUNK_OVERLAP):
    """Chunk by document structure, then split oversized sections with overlap.

    Unlike the old implementation, a missing ``1. DEFINITIONS``-style header
    does not silently convert the whole document into fixed-size windows.
    We try, in order: heading conventions, blank-line paragraphs, single-line
    breaks (for docs with one clause per line and no blank lines), then
    sentence boundaries — only falling through to fixed-size splitting of the
    whole document if none of those find anything at all. Sections that are
    still oversized after this structural pass are split by fixed_chunk,
    which is itself LangChain-backed and separator-aware (see fixed_chunk's
    docstring) rather than a raw arithmetic window.
    """
    boundaries, _method = _structure_boundaries(text)
    chunks = []

    for s, e in zip(boundaries, boundaries[1:]):
        # Trim only surrounding whitespace for cleaner chunks while preserving
        # the exact character offsets used by LegalBench-RAG evaluation.
        raw_s, raw_e = s, e
        while raw_s < raw_e and text[raw_s].isspace():
            raw_s += 1
        while raw_e > raw_s and text[raw_e - 1].isspace():
            raw_e -= 1
        if raw_s >= raw_e:
            continue

        seg = text[raw_s:raw_e]
        if len(seg) <= max_size:
            chunks.append({"text": seg, "start": raw_s, "end": raw_e})
            continue

        # Oversized structural section: fixed-size windows with overlap, but
        # still local to that section so we never cross unrelated sections.
        for c in fixed_chunk(seg, max_size, overlap):
            chunks.append({
                "text": c["text"],
                "start": raw_s + c["start"],
                "end": raw_s + c["end"],
            })

    return chunks


def chunk_documents(documents, chunk_fn):
    """documents: {file_path: full_text}. Returns {file_path: [chunk dicts]},
    tagging each chunk with its file_path so retrieval can span >1 doc."""
    chunks_by_doc = {}
    for file_path, text in documents.items():
        chunks = chunk_fn(text)
        for c in chunks:
            c["file_path"] = file_path
        chunks_by_doc[file_path] = chunks
    return chunks_by_doc
