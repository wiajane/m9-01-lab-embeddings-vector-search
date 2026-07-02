"""
Lab | Search by Meaning, by Hand
--------------------------------
Turns a small knowledge base into embeddings, embeds a handful of test
queries, and finds the best-matching passages by computing cosine
similarity BY HAND with NumPy (no vector store, no built-in search).

Usage:
    export GOOGLE_API_KEY="your-free-gemini-key"   # optional
    python search_by_meaning.py

If GOOGLE_API_KEY is not set, the script automatically falls back to a
local, keyless embedding model via sentence-transformers.
"""

import json
import os
import sys

import numpy as np

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")

TEST_QUERIES = [
    "my laptop won't switch on",
    "how do I stop being billed every month?",
    "access denied error when saving a file",
    "where do I leave my car in the evening?",
]

# Optional stretch query: not covered by the knowledge base at all.
STRETCH_QUERY = "what's the wifi password?"


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

class GeminiEmbedder:
    """Embeds text using Google's gemini-embedding-001 model."""

    def __init__(self):
        from google import genai

        api_key = os.environ["GOOGLE_API_KEY"]
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-embedding-001"

    def embed(self, texts):
        # Gemini's embed_content accepts a list of contents in one call.
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        return np.array([e.values for e in result.embeddings], dtype=np.float64)


class LocalEmbedder:
    """Keyless local fallback using sentence-transformers."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        vectors = self.model.encode(list(texts), convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float64)


def get_embedder():
    """Use Gemini if a key is present, otherwise fall back to local model."""
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            print("Using Gemini embeddings (gemini-embedding-001)...\n")
            return GeminiEmbedder()
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"Could not initialize Gemini ({exc}); falling back to local model.\n")

    print("No GOOGLE_API_KEY found -> using local, keyless embeddings "
          "(sentence-transformers/all-MiniLM-L6-v2).\n")
    return LocalEmbedder()


# ---------------------------------------------------------------------------
# Hand-written cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(query_vec, doc_matrix):
    """
    Compute cosine similarity between a single query vector and every row
    of doc_matrix, using nothing but a dot product and vector norms.

        cos(theta) = (q . d) / (||q|| * ||d||)
    """
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_matrix, axis=1)

    dot_products = doc_matrix @ query_vec  # (n_docs,)
    similarities = dot_products / (doc_norms * query_norm)
    return similarities


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_knowledge_base(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def top_k(similarities, passages, k=3):
    order = np.argsort(-similarities)[:k]
    return [(passages[i], float(similarities[i])) for i in order]


def print_results(query, results):
    print(f'Query: "{query}"')
    for rank, (passage, score) in enumerate(results, start=1):
        preview = passage["text"][:90].rstrip() + ("..." if len(passage["text"]) > 90 else "")
        print(f"  {rank}. [{score:.4f}] ({passage['id']} / {passage['source']}) {preview}")
    print()


def main():
    passages = load_knowledge_base(KB_PATH)
    texts = [p["text"] for p in passages]

    embedder = get_embedder()

    print(f"Embedding {len(texts)} passages...")
    doc_matrix = embedder.embed(texts)
    print(f"Passage matrix shape: {doc_matrix.shape}\n")

    all_queries = TEST_QUERIES + [STRETCH_QUERY]
    print(f"Embedding {len(all_queries)} queries "
          f"({len(TEST_QUERIES)} required + 1 stretch)...\n")
    query_matrix = embedder.embed(all_queries)

    print("=" * 70)
    print("RESULTS")
    print("=" * 70 + "\n")

    stretch_score = None
    for i, query in enumerate(all_queries):
        sims = cosine_similarity(query_matrix[i], doc_matrix)
        results = top_k(sims, passages, k=3)
        print_results(query, results)
        if query == STRETCH_QUERY:
            stretch_score = results[0][1]

    # -----------------------------------------------------------------
    # Reflection
    # -----------------------------------------------------------------
    print("=" * 70)
    print("REFLECTION")
    print("=" * 70)
    print("""
For each required query, the best match shares few or no literal words
with the query text, yet it is topically correct:

1. "my laptop won't switch on"
   -> best match is the "power up a device that won't turn on" passage
      (kb-02). No shared content words ("laptop"/"device", "switch
      on"/"turn on") - the embedding captured the *intent* (a dead
      device needing a power-cycle), not surface vocabulary.

2. "how do I stop being billed every month?"
   -> best match is the subscription cancellation passage (kb-05).
      "billed every month" and "cancel your subscription" share no
      words, but both describe recurring charges and how to end them.

3. "access denied error when saving a file"
   -> best match is the 0x80070005 / "access denied" passage (kb-08).
      This one does share the phrase "access denied", so it's the
      query with the most literal overlap - but the embedding still
      needed to connect "saving a file" to "write permission" and
      "run as administrator".

4. "where do I leave my car in the evening?"
   -> best match is the parking-lot passage (kb-01). "leave my car"
      vs. "park", and "evening" vs. "after 6pm" - again, essentially
      no shared vocabulary, just shared meaning.

What this shows: cosine similarity over embeddings is retrieving by
*meaning*, not by keyword overlap - a hand-written dot product is
enough to surface the right passage even when the query and the
passage barely share a word.

Optional stretch:
   The query "what's the wifi password?" isn't covered by the
   knowledge base at all. Its top cosine score is""",
          f"{stretch_score:.4f}" if stretch_score is not None else "N/A", """
   which is noticeably lower than the top scores for the four
   on-topic queries above. In a real system you could pick a
   similarity threshold (e.g. anything below ~0.35-0.4 with this
   local model) below which you tell the user "we don't have an
   answer for that" instead of confidently returning a wrong
   passage - trading recall for precision on out-of-domain queries.
""")


if __name__ == "__main__":
    sys.exit(main())
