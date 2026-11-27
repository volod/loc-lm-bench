"""Parameters of the `correct` stage. Proposals are read only for a `fake:` model."""

from collections.abc import Mapping, Sequence
from typing import Any

from llb.prep.book.correct.proposal import Proposal
from llb.prep.book.models import FAKE_BACKEND
from llb.prep.book.stages.recognize import RECOGNITION_STAGES

_PARAMS = ("primary", "proposals")
_CHOICE = ("stage", "engine", "variant", "model")
_PROPOSAL = ("text", "start", "end", "replacement", "class", "evidence", "visual", "page_id")
_STAGES = ", ".join(RECOGNITION_STAGES)


def validate_correct_params(params: Mapping[str, Any]) -> list[str]:
    """Problems with the configured primary reading and any scripted proposals."""
    errors = [f"correct: unknown parameter {key!r}" for key in params if key not in _PARAMS]
    primary = params.get("primary")
    if not isinstance(primary, dict):
        errors.append("correct: primary must name the reading a page shows")
    else:
        errors.extend(_choice_errors(primary))
    proposals = params.get("proposals", [])
    if not isinstance(proposals, list):
        errors.append("correct: proposals must be a list")
    else:
        for index, item in enumerate(proposals):
            errors.extend(_proposal_errors(item, index))
    return errors


def validate_correct_model(model: str | None, params: Mapping[str, Any]) -> list[str]:
    """Scripted proposals are the fake model. A served model must not be handed a script."""
    proposals = params.get("proposals")
    if not proposals:
        return []
    backend = (model or "").partition(":")[0]
    if backend == FAKE_BACKEND:
        return []
    return ["correct: proposals are only read for a fake: model"]


def order_errors(params: Mapping[str, Any], configured_before: Sequence[str]) -> list[str]:
    """The primary reading's stage has to have run, or the block will not carry it."""
    primary = params.get("primary")
    if not isinstance(primary, dict):
        return []
    stage = primary.get("stage")
    if stage in RECOGNITION_STAGES and stage not in configured_before:
        return [f"stage 'correct' reads the {stage!r} reading, which must be configured before it"]
    return []


def flat_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """The primary reading and the scripted proposals, with defaults filled in."""
    primary = params["primary"]
    model = primary.get("model")
    return {
        "stage": str(primary["stage"]),
        "engine": str(primary["engine"]),
        "variant": str(primary.get("variant") or "original"),
        "model": None if not model else str(model),
        "proposals": list(params.get("proposals") or []),
    }


def scripted_proposals(params: Mapping[str, Any], page_id: str, text: str) -> list[Proposal]:
    """Proposals whose `text` is this block and whose page, when set, is this page."""
    found: list[Proposal] = []
    for item in params.get("proposals") or []:
        if item.get("text") != text:
            continue
        named = item.get("page_id")
        if isinstance(named, str) and named and named != page_id:
            continue
        found.append(
            Proposal(
                start=item["start"],
                end=item["end"],
                replacement=str(item["replacement"]),
                confusion_class=str(item["class"]),
                evidence=str(item.get("evidence") or ""),
                visual=bool(item.get("visual")),
            )
        )
    return found


def _choice_errors(choice: Mapping[str, Any]) -> list[str]:
    errors = [f"correct primary: unknown parameter {key!r}" for key in choice if key not in _CHOICE]
    if choice.get("stage") not in RECOGNITION_STAGES:
        errors.append(f"correct primary: stage must be one of {_STAGES}")
    engine = choice.get("engine")
    if not isinstance(engine, str) or not engine:
        errors.append("correct primary: engine must name the primary reading")
    variant = choice.get("variant", "original")
    if not isinstance(variant, str) or not variant:
        errors.append("correct primary: variant must be a non-empty string")
    model = choice.get("model")
    if model is not None and (not isinstance(model, str) or not model):
        errors.append("correct primary: model must be a non-empty string when set")
    return errors


def _proposal_errors(item: object, index: int) -> list[str]:
    where = f"correct proposals[{index}]"
    if not isinstance(item, dict):
        return [f"{where} must be a mapping"]
    errors = [f"{where}: unknown parameter {key!r}" for key in item if key not in _PROPOSAL]
    if not isinstance(item.get("text"), str):
        errors.append(f"{where}: text must be the block's primary reading")
    if type(item.get("start")) is not int or type(item.get("end")) is not int:
        errors.append(f"{where}: start and end must be character offsets")
    if not isinstance(item.get("replacement"), str) or not isinstance(item.get("class"), str):
        errors.append(f"{where}: replacement and class must be strings")
    if not isinstance(item.get("evidence"), str):
        errors.append(f"{where}: evidence must be a string")
    if type(item.get("visual")) is not bool:
        errors.append(f"{where}: visual must be true or false")
    page_id = item.get("page_id")
    if page_id is not None and not isinstance(page_id, str):
        errors.append(f"{where}: page_id must be a string when set")
    return errors
