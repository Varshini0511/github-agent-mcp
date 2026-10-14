"""Unit tests for retrieval.py (BM25 + Reciprocal Rank Fusion). Pure Python."""

from retrieval import BM25, reciprocal_rank_fusion, tokenize

CORPUS = [
    "def get_policies_async(policy_type): return rows filtered by policy_type",  # 0
    "Authentication middleware validates the JWT bearer token on every request",  # 1
    "Rate limiting uses exponential backoff with jitter when GitHub returns 429",  # 2
    "class PolicyRow: id, policy_text, embedded flag for the vector index",  # 3
    "README: install dependencies and run the streamlit app",  # 4
]


def test_tokenizer_splits_snake_and_camel_case():
    assert tokenize("getPoliciesAsync") == ["get", "policies", "async"]
    assert tokenize("get_policies_async") == ["get", "policies", "async"]
    assert tokenize("HTTPServer") == ["httpserver"] or "http" in tokenize("HTTPServer")


def test_tokenizer_drops_stopwords_and_lowercases():
    assert tokenize("The Quick brown FOX") == ["quick", "brown", "fox"]


def test_bm25_finds_an_exact_identifier():
    bm = BM25(CORPUS)
    assert bm.top("get_policies_async", 3)[0] == 0


def test_bm25_ranks_the_more_relevant_document_first():
    bm = BM25(CORPUS)
    assert bm.top("exponential backoff jitter 429", 3)[0] == 2
    assert bm.top("JWT token validation", 3)[0] == 1


def test_bm25_excludes_documents_with_no_matching_terms():
    bm = BM25(CORPUS)
    assert bm.top("kubernetes helm chart", 5) == []


def test_bm25_handles_empty_inputs():
    assert BM25([]).top("anything", 3) == []
    assert BM25(CORPUS).top("", 3) == []


def test_rrf_rewards_agreement_between_rankers():
    dense = ["a", "b", "c", "d"]
    lexical = ["c", "a", "x", "y"]
    fused = reciprocal_rank_fusion([dense, lexical])
    # a and c appear in both lists -> they outrank items only one list found.
    assert set(fused[:2]) == {"a", "c"}
    assert fused.index("a") < fused.index("b")
    assert fused.index("c") < fused.index("d")


def test_rrf_recovers_a_keyword_hit_dense_search_missed():
    """The reason hybrid exists: dense never returned 'exact', lexical did."""
    dense = ["m1", "m2", "m3"]
    lexical = ["exact", "m1"]
    fused = reciprocal_rank_fusion([dense, lexical], top_n=3)
    assert "exact" in fused


def test_rrf_respects_top_n_and_is_stable():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "b", "c"]], top_n=2)
    assert fused == ["a", "b"]
    assert reciprocal_rank_fusion([]) == []


def test_looks_like_identifier_distinguishes_code_from_prose():
    from retrieval import looks_like_identifier

    for q in [
        "get_policies_async",
        "getPoliciesAsync",
        "Policies.tsx",
        "src/app/page.tsx",
        "E1042",
        "init()",
    ]:
        assert looks_like_identifier(q), q
    for q in ["how is authentication handled", "where do we retry on rate limits"]:
        assert not looks_like_identifier(q), q


def test_weights_break_ties_toward_the_trusted_retriever():
    dense = ["d1", "d2"]
    lexical = ["exact"]
    assert reciprocal_rank_fusion([dense, lexical])[0] == "d1"  # tie -> first list
    assert reciprocal_rank_fusion([dense, lexical], weights=[1.0, 2.0])[0] == "exact"
