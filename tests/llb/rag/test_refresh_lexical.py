"""Incremental BM25 merge == LexicalIndex.build over the merged texts (exact equivalence)."""

from llb.rag.lexical import LexicalIndex
from llb.rag.refresh.lexical_merge import invert_postings, merge_lexical_index

OLD_TEXTS = [
    "Тарас Шевченко написав Кобзар",
    "Іван Франко написав Мойсей",
    "Леся Українка написала Лісову пісню",
]


def test_invert_postings_recovers_exact_term_counts():
    index = LexicalIndex.build(["a b b c", "b c"])
    per_ordinal = invert_postings(index)
    assert per_ordinal == [{"a": 1, "b": 2, "c": 1}, {"b": 1, "c": 1}]


def test_merge_matches_build_for_add_modify_delete():
    old = LexicalIndex.build(OLD_TEXTS)
    # keep 0, replace 1 (modified doc), drop 2 (deleted doc), append a new text (added doc)
    new_text_b = "Іван Франко написав Захара Беркута"
    new_text_d = "Михайло Коцюбинський написав Тіні забутих предків"
    merged = merge_lexical_index(old, [0, new_text_b, new_text_d])
    rebuilt = LexicalIndex.build([OLD_TEXTS[0], new_text_b, new_text_d])
    assert merged.postings == rebuilt.postings
    assert merged.doc_lengths == rebuilt.doc_lengths
    assert merged.lemmatize == rebuilt.lemmatize
    # the deleted doc's unique term is gone from the merged postings
    assert "українка" not in merged.postings
    # identical BM25 ranking on a probe query
    assert merged.search("написав Франко", 3) == rebuilt.search("написав Франко", 3)


def test_merge_reorders_kept_ordinals():
    old = LexicalIndex.build(OLD_TEXTS)
    merged = merge_lexical_index(old, [2, 0])  # doc deletion shifts every ordinal
    rebuilt = LexicalIndex.build([OLD_TEXTS[2], OLD_TEXTS[0]])
    assert merged.postings == rebuilt.postings
    assert merged.doc_lengths == rebuilt.doc_lengths


def test_merge_lemmatized_tokenizes_only_new_texts():
    calls: list[str] = []

    def lemmatizer(token: str) -> str:
        calls.append(token)
        return token[:4]

    old = LexicalIndex.build(OLD_TEXTS, lemmatize=True, lemmatizer=lemmatizer)
    calls.clear()
    new_text = "Григорій Сковорода мандрував Україною"
    merged = merge_lexical_index(old, [0, 1, new_text], lemmatizer=lemmatizer)
    assert set(calls) == {"григорій", "сковорода", "мандрував", "україною"}
    rebuilt = LexicalIndex.build(
        [OLD_TEXTS[0], OLD_TEXTS[1], new_text], lemmatize=True, lemmatizer=lemmatizer
    )
    assert merged.postings == rebuilt.postings
    assert merged.doc_lengths == rebuilt.doc_lengths
