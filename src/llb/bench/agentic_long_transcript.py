"""Multi-step agentic tasks with MEDIUM observations -- the keep_last_n long-transcript shape.

Search-count / search-locate dump 20k-36k-char observations, so a 6-step budget overflows before
the transcript ever grows past `keep_last_n=3`. This module builds the complementary shape:

  * medium-observation search tasks -- shrink a real count/locate set so one search fits the window
    while max_steps > keep lets the kept window differ (the CUDA evidence path);
  * synthetic file/db pipelines -- for CI over a fake complete (live models may loop on read_file).
"""

import copy
import re
from typing import Any

from llb.bench.agentic.model import ASSERT_ANSWER_CONTAINS, ASSERT_DB_EQUALS, ASSERT_FILE_EQUALS
from llb.bench.agentic_tasks import _doc_count, _docs_containing

# Intended path length sits ABOVE the shipped keep (3) so older steps are actually dropped.
DEFAULT_PIPELINE_DEPTH = 4
# Char padding per planted file so each observation is medium, not a one-line stub, and a full
# transcript of depth steps still fits a ~20k-char prompt budget.
DEFAULT_OBS_PAD_CHARS = 160
# Step budget for the long-transcript keep grid: depth reads + depth writes, with headroom.
DEFAULT_LONG_TRANSCRIPT_MAX_STEPS = 12
# Medium-search defaults: ~6 matching + 6 other docs * 180 chars ~= 2k-char observations.
DEFAULT_MAX_MATCH_DOCS = 6
DEFAULT_MAX_OTHER_DOCS = 6
DEFAULT_MAX_DOC_CHARS = 180

_PAD_UNIT = "записка про етап обробки даних громади "
_QUERY_IN_PROMPTS = re.compile(r"«([^»]+)»")


def _padded(payload: str, pad_chars: int) -> str:
    """Payload plus deterministic UA filler so the observation has real bulk."""
    if pad_chars <= 0:
        return payload
    filler = (_PAD_UNIT * ((pad_chars // len(_PAD_UNIT)) + 1))[:pad_chars]
    return f"{payload}\n{filler}"


def pipeline_db_task(
    index: int,
    *,
    depth: int = DEFAULT_PIPELINE_DEPTH,
    pad_chars: int = DEFAULT_OBS_PAD_CHARS,
    values: list[str] | None = None,
) -> dict[str, Any]:
    """Read depth files and store each first line in the DB under ``k{i}``.

    Intended path is ``depth`` reads + ``depth`` ``db_set`` calls (2*depth steps). Early values
    live in the world, so keep_last_n dropping them from the prompt does not lose the answer.
    """
    if depth < 2:
        raise ValueError(f"pipeline depth must be >= 2, got {depth}")
    tokens = values if values is not None else [f"v{index}-{i}" for i in range(depth)]
    if len(tokens) != depth:
        raise ValueError(f"values length {len(tokens)} != depth {depth}")
    files = {f"n{i}.txt": _padded(tokens[i], pad_chars) for i in range(depth)}
    steps = "; ".join(
        f"{i + 1}) прочитай n{i}.txt і збережи перший рядок у базі під ключем k{i}"
        for i in range(depth)
    )
    return {
        "id": f"pipeline-db-{index:03d}-d{depth}",
        "prompt": (
            f"Виконай кроки по порядку. {steps}. "
            f"Не викликай finish, доки всі ключі k0..k{depth - 1} не записані."
        ),
        "setup": {"files": files},
        "success": [
            {"kind": ASSERT_DB_EQUALS, "key": f"k{i}", "value": tokens[i]} for i in range(depth)
        ],
    }


def pipeline_sum_task(
    index: int,
    *,
    depth: int = DEFAULT_PIPELINE_DEPTH,
    pad_chars: int = DEFAULT_OBS_PAD_CHARS,
    values: list[int] | None = None,
) -> dict[str, Any]:
    """Read depth files into the DB, then write their sum to ``sum.txt``.

    Intermediate numbers are stored under ``k{i}`` so a keep=1 prompt can still finish the sum
    by re-reading the DB rather than relying on dropped transcript lines.
    """
    if depth < 2:
        raise ValueError(f"pipeline depth must be >= 2, got {depth}")
    nums = values if values is not None else [((index + i) % 9) + 1 for i in range(depth)]
    if len(nums) != depth:
        raise ValueError(f"values length {len(nums)} != depth {depth}")
    files = {f"n{i}.txt": _padded(str(nums[i]), pad_chars) for i in range(depth)}
    total = sum(nums)
    store_steps = "; ".join(
        f"{i + 1}) прочитай n{i}.txt і збережи перше число у базі під ключем k{i}"
        for i in range(depth)
    )
    return {
        "id": f"pipeline-sum-{index:03d}-d{depth}",
        "prompt": (
            f"Виконай кроки по порядку. {store_steps}; "
            f"{depth + 1}) через calculator додай значення k0..k{depth - 1} "
            f"(спочатку db_get кожного ключа) і запиши лише суму у файл sum.txt."
        ),
        "setup": {"files": files},
        "success": [
            *[
                {"kind": ASSERT_DB_EQUALS, "key": f"k{i}", "value": str(nums[i])}
                for i in range(depth)
            ],
            {"kind": ASSERT_FILE_EQUALS, "path": "sum.txt", "value": str(total)},
        ],
    }


def pipeline_copy_task(
    index: int,
    *,
    depth: int = DEFAULT_PIPELINE_DEPTH,
    pad_chars: int = DEFAULT_OBS_PAD_CHARS,
) -> dict[str, Any]:
    """Read depth source files and write each first-line token into ``out{i}.txt``.

    One write per file keeps progress in the world; success checks every out file.
    """
    if depth < 2:
        raise ValueError(f"pipeline depth must be >= 2, got {depth}")
    tokens = [f"t{index}-{i}" for i in range(depth)]
    files = {f"src{i}.txt": _padded(tokens[i], pad_chars) for i in range(depth)}
    steps = "; ".join(
        f"{i + 1}) прочитай src{i}.txt і запиши перший рядок у файл out{i}.txt"
        for i in range(depth)
    )
    return {
        "id": f"pipeline-copy-{index:03d}-d{depth}",
        "prompt": f"Виконай кроки по порядку. {steps}.",
        "setup": {"files": files},
        "success": [
            {"kind": ASSERT_FILE_EQUALS, "path": f"out{i}.txt", "value": tokens[i]}
            for i in range(depth)
        ],
    }


def _query_from_prompt(prompt: str) -> str | None:
    """Extract the «query» planted in the search_count / search_locate prompt templates."""
    match = _QUERY_IN_PROMPTS.search(prompt)
    return match.group(1) if match else None


def medium_observation_search_task(
    task: dict[str, Any],
    *,
    max_match_docs: int = DEFAULT_MAX_MATCH_DOCS,
    max_other_docs: int = DEFAULT_MAX_OTHER_DOCS,
    max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
) -> dict[str, Any] | None:
    """Shrink a count/locate task so one search observation fits the keep window.

    Keeps a bounded number of matching + non-matching docs (truncated), and REBINDS the success
    assertion from the shrunk corpus so the objective answer stays correct.
    """
    tid = str(task.get("id", ""))
    corpus = (task.get("setup") or {}).get("corpus")
    if not isinstance(corpus, dict):
        return None
    query = _query_from_prompt(str(task.get("prompt", "")))
    if not query:
        return None
    matching = [doc_id for doc_id, text in corpus.items() if query.casefold() in text.casefold()]
    other = [doc_id for doc_id in corpus if doc_id not in set(matching)]
    kept_ids = matching[:max_match_docs] + other[:max_other_docs]
    if not kept_ids:
        return None
    shrunk = {doc_id: str(corpus[doc_id])[:max_doc_chars] for doc_id in kept_ids}
    out = copy.deepcopy(task)
    out["setup"] = {"corpus": shrunk}
    out["id"] = f"medium-{tid}"
    if tid.startswith("search-count") or "count" in tid:
        out["success"] = [{"kind": ASSERT_ANSWER_CONTAINS, "value": str(_doc_count(shrunk, query))}]
        return out
    if tid.startswith("search-locate") or "locate" in tid:
        hits = _docs_containing(shrunk, query)
        if len(hits) != 1:
            # Truncation or hit-cap removed the unique match -- drop the task.
            return None
        out["success"] = [{"kind": ASSERT_ANSWER_CONTAINS, "value": hits[0]}]
        return out
    return None


def build_long_transcript_from_search_tasks(
    search_tasks: list[dict[str, Any]],
    *,
    max_match_docs: int = DEFAULT_MAX_MATCH_DOCS,
    max_other_docs: int = DEFAULT_MAX_OTHER_DOCS,
    max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
) -> list[dict[str, Any]]:
    """Turn fat-observation search tasks into medium-observation long-transcript tasks."""
    out: list[dict[str, Any]] = []
    for task in search_tasks:
        built = medium_observation_search_task(
            task,
            max_match_docs=max_match_docs,
            max_other_docs=max_other_docs,
            max_doc_chars=max_doc_chars,
        )
        if built is not None:
            out.append(built)
    return out


def build_long_transcript_tasks(
    *,
    n_db: int = 8,
    n_copy: int = 6,
    n_sum: int = 6,
    depth: int = DEFAULT_PIPELINE_DEPTH,
    pad_chars: int = DEFAULT_OBS_PAD_CHARS,
) -> list[dict[str, Any]]:
    """Deterministic multi-step file/db tasks (unit-test shape; live models may loop on reads)."""
    tasks: list[dict[str, Any]] = []
    for i in range(n_db):
        d = depth + (i % 2)
        tasks.append(pipeline_db_task(i, depth=d, pad_chars=pad_chars))
    for i in range(n_copy):
        d = depth + (i % 2)
        tasks.append(pipeline_copy_task(i, depth=d, pad_chars=pad_chars))
    for i in range(n_sum):
        tasks.append(pipeline_sum_task(i, depth=depth, pad_chars=pad_chars))
    return tasks
