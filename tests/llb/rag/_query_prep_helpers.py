"""Shared fixtures for focused query-preparation tests."""

from llb.rag.query_prep.glossary import Glossary, GlossaryEntry


def glossary() -> Glossary:
    return Glossary(
        (
            GlossaryEntry("інтелектуальна власність", ("ІВ", "intelektualna vlasnist")),
            GlossaryEntry("авторське право", ()),
        )
    )


class RecordingStore:
    def __init__(self, chunks):
        self.chunks = chunks
        self.seen: list[str] = []

    def retrieve(self, question, k):
        self.seen.append(question)
        return self.chunks[:k]
