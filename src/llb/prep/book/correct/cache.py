"""Per-block proposal cache. A page re-run asks the model only about blocks whose key changed."""

import hashlib
import json
from pathlib import Path

from llb.prep.book.cache import digest_json
from llb.prep.book.correct.constants import CACHE_DIRNAME, PROMPT_VERSION, guard_identity
from llb.prep.book.correct.proposal import Proposal, ProposalResult


def cache_key(
    model: str,
    anchor: str,
    crop_png: bytes,
    text: str,
    before: str,
    after: str,
    previous_page: str,
    next_page: str,
) -> str:
    """Digest of everything the model sees for one block, plus the guard and prompt versions."""
    return digest_json(
        {
            "model": model,
            "guard": guard_identity(),
            "prompt": PROMPT_VERSION,
            "anchor": anchor,
            "crop": hashlib.sha256(crop_png).hexdigest(),
            "text": text,
            "before": before,
            "after": after,
            "previous_page": previous_page,
            "next_page": next_page,
        }
    )


def read_cache(directory: Path, block_id: str, key: str) -> ProposalResult | None:
    """The stored reply when the key still matches, else None."""
    path = directory / CACHE_DIRNAME / f"{block_id}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("key") != key:
        return None
    proposals = payload.get("proposals")
    if not isinstance(proposals, list):
        return None
    parsed: list[Proposal] = []
    for item in proposals:
        proposal = _proposal(item)
        if proposal is None:
            return None
        parsed.append(proposal)
    return ProposalResult(tuple(parsed), unparsed=bool(payload.get("unparsed")))


def write_cache(directory: Path, block_id: str, key: str, result: ProposalResult) -> None:
    """Store one reply under the stage work directory. The commit keeps it with the page."""
    path = directory / CACHE_DIRNAME / f"{block_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": key,
        "unparsed": result.unparsed,
        "proposals": [_dump(item) for item in result.proposals],
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _proposal(item: object) -> Proposal | None:
    if not isinstance(item, dict):
        return None
    start, end = item.get("start"), item.get("end")
    replacement, confusion_class = item.get("replacement"), item.get("class")
    evidence, visual = item.get("evidence"), item.get("visual")
    if type(start) is not int or type(end) is not int:
        return None
    if not isinstance(replacement, str) or not isinstance(confusion_class, str):
        return None
    if not isinstance(evidence, str) or type(visual) is not bool:
        return None
    return Proposal(start, end, replacement, confusion_class, evidence, visual)


def _dump(proposal: Proposal) -> dict[str, object]:
    return {
        "start": proposal.start,
        "end": proposal.end,
        "replacement": proposal.replacement,
        "class": proposal.confusion_class,
        "evidence": proposal.evidence,
        "visual": proposal.visual,
    }
