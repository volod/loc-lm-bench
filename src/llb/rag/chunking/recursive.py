"""The pinned langchain `RecursiveCharacterTextSplitter` lane (offset-exact, version-locked).

Chunk boundaries are part of the index contract, so the recursive splitter is a pinned base
dependency (see `dependencies` in pyproject.toml). A missing or version-drifted install fails
loudly here instead of silently rechunking.
"""

from typing import Any

_REQUIRED_TEXT_SPLITTERS = "1.1.2"
_recursive_splitter_cls: Any = None


def _require_recursive_splitter() -> Any:
    """Return the pinned `RecursiveCharacterTextSplitter`, failing early on a bad install."""
    global _recursive_splitter_cls
    if _recursive_splitter_cls is not None:
        return _recursive_splitter_cls
    try:
        from importlib.metadata import version

        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:  # a required base dependency is missing
        raise RuntimeError(
            "recursive/markdown chunking requires `langchain-text-splitters"
            f"=={_REQUIRED_TEXT_SPLITTERS}` (a base dependency); reinstall with "
            "`uv pip install -e .`."
        ) from exc
    found = version("langchain-text-splitters")
    if found != _REQUIRED_TEXT_SPLITTERS:
        raise RuntimeError(
            f"langchain-text-splitters {found} is installed, but chunk boundaries are pinned to "
            f"{_REQUIRED_TEXT_SPLITTERS}. Reinstall the pinned version so indexes stay reproducible."
        )
    _recursive_splitter_cls = RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter


def recursive_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Offset-exact spans from the pinned langchain `RecursiveCharacterTextSplitter`.

    Every span is verified to reproduce its exact source slice, so a splitter that ever emits a
    non-slice chunk raises here rather than letting misaligned offsets reach the index.
    """
    splitter = _require_recursive_splitter()(
        chunk_size=size, chunk_overlap=overlap, add_start_index=True
    )
    spans: list[tuple[int, int]] = []
    for doc in splitter.create_documents([text]):
        content = doc.page_content
        start = doc.metadata["start_index"]
        end = start + len(content)
        if text[start:end] != content:
            raise ValueError(
                "recursive splitter produced a chunk that is not an exact source slice; "
                "refusing to index misaligned offsets."
            )
        spans.append((start, end))
    return spans
