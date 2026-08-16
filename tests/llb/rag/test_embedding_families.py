"""Every registered embedder convention against the format its model card documents.

These are the tests that make the bake-off readable: scoring a retrieval-tuned encoder under the
wrong input format silently caps its recall and the row just looks bad, so each family's exact
prefix string is asserted here with the card it came from named in the comment.
"""

import pytest

from llb.rag.embedding_families import (
    BGE_QUERY_INSTRUCTION,
    CONVENTIONS,
    FAMILY_BGE,
    FAMILY_BGE_M3,
    FAMILY_E5,
    FAMILY_E5_INSTRUCT,
    FAMILY_GTE_MULTILINGUAL,
    FAMILY_JINA_V3,
    FAMILY_PLAIN,
    FAMILY_QWEN3_EMBEDDING,
    FAMILY_UNKNOWN,
    RETRIEVAL_TASK,
    apply_passage_convention,
    apply_query_convention,
    embedding_family,
    is_registered,
    resolve_convention,
)

Q = "коли набирає чинності договір"
P = "Договір набирає чинності з дня його підписання."


@pytest.mark.parametrize(
    "model, family",
    [
        ("intfloat/multilingual-e5-small", FAMILY_E5),
        ("intfloat/multilingual-e5-base", FAMILY_E5),
        ("intfloat/multilingual-e5-large", FAMILY_E5),
        # The instruct sibling shares the `e5` id stem; resolving it to plain `e5` would score it
        # with `query:` / `passage:` prefixes its card never documents.
        ("intfloat/multilingual-e5-large-instruct", FAMILY_E5_INSTRUCT),
        ("BAAI/bge-m3", FAMILY_BGE_M3),
        ("BAAI/bge-large-en-v1.5", FAMILY_BGE),
        ("Alibaba-NLP/gte-multilingual-base", FAMILY_GTE_MULTILINGUAL),
        ("jinaai/jina-embeddings-v3", FAMILY_JINA_V3),
        ("Qwen/Qwen3-Embedding-0.6B", FAMILY_QWEN3_EMBEDDING),
        ("lang-uk/ukr-paraphrase-multilingual-mpnet-base", FAMILY_PLAIN),
    ],
)
def test_embedding_family_resolves_per_model(model, family):
    assert embedding_family(model) == family
    assert resolve_convention(model).family == family
    assert is_registered(model)


def test_resolution_is_case_insensitive():
    assert embedding_family("INTFLOAT/Multilingual-E5-Base") == FAMILY_E5
    assert embedding_family("QWEN/QWEN3-EMBEDDING-0.6B") == FAMILY_QWEN3_EMBEDDING


# --- the family table's whole purpose: an unknown id must NOT read as `plain` -------------------


@pytest.mark.parametrize(
    "model",
    [
        "some-vendor/never-registered-encoder",
        "acme/multilingual-retriever-v9",
        "",
    ],
)
def test_unregistered_id_resolves_to_unknown_not_plain(model):
    # `plain` is a DOCUMENTED property of the paraphrase/STS line, not a safe default: falling
    # through to it is exactly the silent no-instruction pass this registry exists to prevent.
    assert embedding_family(model) == FAMILY_UNKNOWN
    assert embedding_family(model) != FAMILY_PLAIN
    assert not is_registered(model)


def test_every_registered_family_cites_its_model_card():
    for family, convention in CONVENTIONS.items():
        if family == FAMILY_UNKNOWN:
            assert convention.source == ""
            continue
        assert convention.source.startswith("https://huggingface.co/"), family


def test_default_roster_is_fully_registered():
    from llb.rag.embedding_bakeoff_models import DEFAULT_LOCAL_CANDIDATES

    assert DEFAULT_LOCAL_CANDIDATES[0] == "intfloat/multilingual-e5-base"
    unregistered = [m for m in DEFAULT_LOCAL_CANDIDATES if not is_registered(m)]
    assert unregistered == []


@pytest.mark.parametrize(
    "model",
    [
        "intfloat/multilingual-e5-large-instruct",
        "Alibaba-NLP/gte-multilingual-base",
        "jinaai/jina-embeddings-v3",
        "Qwen/Qwen3-Embedding-0.6B",
    ],
)
def test_current_generation_candidates_are_on_the_default_roster(model):
    from llb.rag.embedding_bakeoff_models import DEFAULT_LOCAL_CANDIDATES

    assert model in DEFAULT_LOCAL_CANDIDATES


# --- per-family input formats, each against its card -------------------------------------------


def test_e5_prefixes_both_sides():
    # Card: `query: {q}` / `passage: {p}` (multilingual-e5-{small,base,large}).
    assert apply_query_convention("intfloat/multilingual-e5-base", [Q]) == [f"query: {Q}"]
    assert apply_passage_convention("intfloat/multilingual-e5-base", [P]) == [f"passage: {P}"]


def test_e5_small_uses_same_prefixes_as_base():
    # Sub-base sibling stays on the e5 query/passage convention; wrong family would cap recall.
    assert apply_query_convention("intfloat/multilingual-e5-small", ["коли"]) == ["query: коли"]
    assert apply_passage_convention("intfloat/multilingual-e5-small", ["текст"]) == [
        "passage: текст"
    ]


def test_e5_instruct_uses_the_instruct_query_and_a_BARE_passage():
    # Card `get_detailed_instruct`: f'Instruct: {task}\nQuery: {query}' (note the space after
    # `Query:`), and "No need to add instruction for retrieval documents".
    model = "intfloat/multilingual-e5-large-instruct"
    assert apply_query_convention(model, [Q]) == [f"Instruct: {RETRIEVAL_TASK}\nQuery: {Q}"]
    assert apply_passage_convention(model, [P]) == [P]
    # The regression this family exists for: it must NOT get the plain-e5 prefixes.
    assert not apply_query_convention(model, [Q])[0].startswith("query: ")


def test_bge_m3_uses_no_prefix_on_either_side():
    # BGE-M3's retrieval default is NO instruction; scoring it with e5 prefixes would cap recall.
    assert apply_query_convention("BAAI/bge-m3", ["коли"]) == ["коли"]
    assert apply_passage_convention("BAAI/bge-m3", ["текст"]) == ["текст"]


def test_bge_v15_instructs_query_only():
    assert apply_query_convention("BAAI/bge-large-en-v1.5", ["q"]) == [BGE_QUERY_INSTRUCTION + "q"]
    assert apply_passage_convention("BAAI/bge-large-en-v1.5", ["p"]) == ["p"]  # passage untouched


def test_gte_multilingual_is_symmetric_and_needs_remote_code():
    # Card's sentence-transformers usage encodes queries and documents in one bare
    # `model.encode(input_texts)` call, loaded with trust_remote_code=True.
    model = "Alibaba-NLP/gte-multilingual-base"
    assert apply_query_convention(model, [Q]) == [Q]
    assert apply_passage_convention(model, [P]) == [P]
    convention = resolve_convention(model)
    assert convention.symmetric
    assert convention.trust_remote_code


def test_jina_v3_uses_its_retrieval_prompts_and_task_adapters():
    # Prefixes are the repo's config_sentence_transformers.json prompts; `task=` additionally
    # selects the LoRA adapter, which the prompt text alone does not do.
    model = "jinaai/jina-embeddings-v3"
    assert apply_query_convention(model, [Q]) == [
        f"Represent the query for retrieving evidence documents: {Q}"
    ]
    assert apply_passage_convention(model, [P]) == [f"Represent the document for retrieval: {P}"]
    convention = resolve_convention(model)
    assert convention.query_kwargs == {"task": "retrieval.query"}
    assert convention.passage_kwargs == {"task": "retrieval.passage"}
    assert convention.trust_remote_code


def test_qwen3_embedding_uses_the_instruct_query_prompt_with_no_trailing_space():
    # Repo's config_sentence_transformers.json "query" prompt, verbatim: it ends at `Query:` with
    # NO trailing space (unlike e5-instruct), and the "document" prompt is the empty string.
    model = "Qwen/Qwen3-Embedding-0.6B"
    assert apply_query_convention(model, [Q]) == [f"Instruct: {RETRIEVAL_TASK}\nQuery:{Q}"]
    assert apply_passage_convention(model, [P]) == [P]
    assert not resolve_convention(model).trust_remote_code


def test_plain_paraphrase_model_is_symmetric_no_prefix():
    model = "lang-uk/ukr-paraphrase-multilingual-mpnet-base"
    assert apply_query_convention(model, ["q"]) == ["q"]
    assert apply_passage_convention(model, ["p"]) == ["p"]
    assert resolve_convention(model).symmetric


def test_unknown_family_applies_no_instruction_and_carries_no_kwargs():
    model = "some-vendor/never-registered-encoder"
    assert apply_query_convention(model, [Q]) == [Q]
    assert apply_passage_convention(model, [P]) == [P]
    convention = resolve_convention(model)
    assert convention.query_kwargs == {} and convention.passage_kwargs == {}
    assert not convention.trust_remote_code
