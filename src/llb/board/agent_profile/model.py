"""Vocabulary of the composed agent operating profile: the field roster, the four field states, the
dependency axes a demotion travels along, and the record dataclasses.

An operating profile is ONE artifact assembled from lanes that each measured a different knob. The
whole failure mode it invites is a default dressed up as a recommendation, so every field carries
where its value came from, what that lane read, and how old the reading is -- and a field whose lane
never ran is `unmeasured` rather than filled in.

A leaf module: `sources_rag` / `sources_agent` (per-lane readings), `compose` (the guard and the
demotion), `render` (payload + rationale), and `replay` (the flags) build on it.
"""

from dataclasses import dataclass, field

from llb.core.contracts.common import JsonObject

METHOD = "agent-profile"
PROFILE_JSON = "agent_profile.json"
PROFILE_MD = "profile.md"

# The four states a field can be in. `measured` is the only one an operator may act on.
STATE_MEASURED = "measured"
# The lane never ran on this host: no value, and never a default standing in for one.
STATE_UNMEASURED = "unmeasured"
# A value exists but something it rests on moved (a stale adapter, a changed retrieval knob).
STATE_DEMOTED = "demoted"
# The value was measured against a different corpus, store, or model than the profile anchors on;
# mixing it in would silently compose a configuration nobody ever ran.
STATE_REFUSED = "refused"

# What a field's reading rests on. A demotion travels along these axes and no others.
DEP_STORE = "store"
DEP_ADAPTER = "adapter"

FIELD_MODEL = "model"
FIELD_BACKEND = "backend"
FIELD_PROMPT_SYSTEM = "prompt_system_id"
FIELD_ADAPTER = "adapter"
FIELD_CONTEXT_POLICY = "context_policy"
FIELD_CONTEXT_ORDER = "context_order"
FIELD_TOP_K = "top_k"
FIELD_RERANKER = "reranker"
FIELD_CONTEXT_BUDGET = "context_budget"
FIELD_LOOP_POLICY = "loop_policy"

# The consistency axes a field is checked on. `model` is checked only for lanes that ran a model.
KEY_MODEL = "model"
KEY_CORPUS = "corpus_root"
KEY_STORE = "retrieval_fingerprint"


@dataclass(frozen=True)
class FieldSpec:
    """One profile field: which lane measures it and what its reading rests on."""

    name: str
    lane: str
    depends: frozenset[str]
    summary: str


PROFILE_FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        FIELD_MODEL,
        "run-eval",
        frozenset({DEP_STORE, DEP_ADAPTER}),
        "the model to serve, host-fit and Pareto-optimal on the ranked cohort",
    ),
    FieldSpec(
        FIELD_BACKEND,
        "run-eval",
        frozenset({DEP_STORE, DEP_ADAPTER}),
        "the runtime the recommended model was measured on",
    ),
    FieldSpec(
        FIELD_PROMPT_SYSTEM,
        "prompt-system-compare",
        frozenset({DEP_STORE, DEP_ADAPTER}),
        "the reviewed prompt-system package the model scored best under",
    ),
    FieldSpec(
        FIELD_ADAPTER,
        "adapter-registry",
        frozenset({DEP_ADAPTER}),
        "the registered adapter to load, or none",
    ),
    FieldSpec(
        FIELD_CONTEXT_POLICY,
        "agentic-context",
        frozenset({DEP_ADAPTER}),
        "how the agent spends its context window as the transcript grows",
    ),
    FieldSpec(
        FIELD_CONTEXT_ORDER,
        "context-position",
        frozenset({DEP_STORE, DEP_ADAPTER}),
        "how retrieved chunks are laid into the prompt",
    ),
    FieldSpec(
        FIELD_TOP_K,
        "run-eval",
        frozenset({DEP_STORE}),
        "retrieval depth of the winning configuration cell",
    ),
    FieldSpec(
        FIELD_RERANKER,
        "compare-rerankers",
        frozenset({DEP_STORE}),
        "the cross-encoder the bake-off ranks first, or none",
    ),
    FieldSpec(
        FIELD_CONTEXT_BUDGET,
        "run-eval",
        frozenset(),
        "the per-prompt context budget the winning cell ran under",
    ),
    FieldSpec(
        FIELD_LOOP_POLICY,
        "agentic-loop-policy",
        frozenset({DEP_ADAPTER}),
        "step budget plus malformed-call and repeated-call handling",
    ),
)

FIELD_SPECS_BY_NAME = {spec.name: spec for spec in PROFILE_FIELD_SPECS}


@dataclass
class ProfileField:
    """One recommended value plus everything that makes it checkable.

    `measured_against` records the lane's own corpus / store / model as the artifact stated them --
    that is what the consistency guard compares, and what a refusal names.
    """

    spec: FieldSpec
    value: object | None = None
    state: str = STATE_UNMEASURED
    evidence_path: str | None = None
    verdict: str | None = None
    uncertainty: JsonObject | None = None
    measured_at: str | None = None
    measured_against: JsonObject = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.spec.name

    def note(self, text: str) -> None:
        """Attach one operator-facing reason; duplicates are dropped so a re-check stays quiet."""
        if text not in self.notes:
            self.notes.append(text)

    def demote(self, reason: str) -> None:
        """Keep the value but stop recommending it: something it rests on moved."""
        if self.state == STATE_MEASURED:
            self.state = STATE_DEMOTED
        self.note(reason)

    def refuse(self, reason: str) -> None:
        """Refuse the value outright: it was measured against something else."""
        self.state = STATE_REFUSED
        self.note(reason)


@dataclass
class Anchor:
    """What every field must have been measured against for the composition to be one configuration.

    It comes from the run-eval pick, because that is the only lane that fixes model, corpus, and
    store at once. With no run-eval bundle there is no anchor and nothing can be cross-checked.
    """

    model: str | None = None
    corpus_root: str | None = None
    retrieval_fingerprint: JsonObject | None = None

    @property
    def resolved(self) -> bool:
        return self.model is not None


@dataclass
class AgentProfile:
    """The composed profile: one field per knob, plus the two drift findings that demote them."""

    generated_at: str
    anchor: Anchor
    fields: list[ProfileField]
    store_drift: list[JsonObject] = field(default_factory=list)
    adapter_drift: list[str] = field(default_factory=list)

    def by_name(self, name: str) -> ProfileField:
        return next(item for item in self.fields if item.name == name)

    def measured(self) -> list[ProfileField]:
        return [item for item in self.fields if item.state == STATE_MEASURED]
