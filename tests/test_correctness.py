from llb.scoring import correctness


def test_normalize_strips_punct_and_case():
    assert correctness.normalize("Київ, столиця!") == "київ столиця"


def test_exact_match_ignores_punctuation_and_case():
    assert correctness.exact_match("Київ.", "київ") == 1.0
    assert correctness.exact_match("Львів", "Київ") == 0.0


def test_exact_match_empty_reference_is_zero():
    assert correctness.exact_match("", "") == 0.0


def test_token_f1_partial_overlap():
    f1 = correctness.token_f1("столиця України Київ", "Київ столиця")
    assert 0.0 < f1 < 1.0


def test_token_f1_perfect():
    assert correctness.token_f1("Київ столиця", "столиця Київ") == 1.0


def test_contains_all_reference_tokens():
    # `contains` is exact-token (no stemming): every reference surface form must appear.
    assert correctness.contains("Київ це столиця держави", "Київ столиця") == 1.0
    assert correctness.contains("Львів", "Київ столиця") == 0.0


def test_answer_correctness_bundle():
    out = correctness.answer_correctness("Київ", "Київ")
    assert out["exact"] == 1.0 and out["score"] == out["token_f1"] == 1.0
    assert "semantic" not in out  # no embedder -> no semantic signal


class FakeEmbedder:
    """Maps text -> a (unit) vector; mimics Embedder.encode_queries."""

    def __init__(self, mapping):
        self.mapping = mapping

    def encode_queries(self, texts):
        return [self.mapping[t] for t in texts]


def test_semantic_similarity_identical_is_one():
    emb = FakeEmbedder({"Київ столиця": [1.0, 0.0], "столиця Київ": [1.0, 0.0]})
    assert correctness.semantic_similarity("Київ столиця", "столиця Київ", emb) == 1.0


def test_semantic_similarity_orthogonal_is_zero():
    emb = FakeEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    assert correctness.semantic_similarity("a", "b", emb) == 0.0


def test_semantic_similarity_negative_is_clamped():
    emb = FakeEmbedder({"a": [1.0, 0.0], "b": [-1.0, 0.0]})
    assert correctness.semantic_similarity("a", "b", emb) == 0.0


def test_semantic_similarity_empty_short_circuits():
    assert correctness.semantic_similarity("", "ref", FakeEmbedder({})) == 0.0  # no encode call


def test_answer_correctness_includes_semantic_with_embedder():
    emb = FakeEmbedder({"Київ": [1.0, 0.0]})
    out = correctness.answer_correctness("Київ", "Київ", embedder=emb)
    assert out["semantic"] == 1.0
    assert out["score"] == out["token_f1"]  # headline stays token_f1
