"""Context-policy vocabulary and tunable constants for agent episodes."""

from dataclasses import dataclass

POLICY_FULL = "full"
POLICY_OBSERVATION_CAP = "observation_cap"
POLICY_KEEP_LAST_N = "keep_last_n"
POLICY_COMPACT = "compact"
CONTEXT_POLICIES: tuple[str, ...] = (
    POLICY_FULL,
    POLICY_OBSERVATION_CAP,
    POLICY_KEEP_LAST_N,
    POLICY_COMPACT,
)

# Defaults pinned by the context-policy constant sweep.
DEFAULT_OBSERVATION_CAP_CHARS = 800
OBSERVATION_HEAD_SHARE = 0.6
DEFAULT_KEEP_LAST_N = 3
DEFAULT_COMPACT_SHARE = 0.5
DEFAULT_COMPACT_KEEP_RECENT = 1

# The summary input can follow the full usable window or the compaction trigger.
SUMMARY_INPUT_CAP_WINDOW = "window"
SUMMARY_INPUT_CAP_TRIGGER = "trigger"
SUMMARY_INPUT_CAPS: tuple[str, ...] = (SUMMARY_INPUT_CAP_WINDOW, SUMMARY_INPUT_CAP_TRIGGER)
DEFAULT_SUMMARY_INPUT_CAP = SUMMARY_INPUT_CAP_WINDOW

# HOW that input cap spends its bytes once the folded transcript does not fit it. `head_tail` cuts
# the rendered transcript once, head and tail around one elision marker, so a fact in the middle
# ENTRIES is gone; `per_entry_head` shares the same byte budget across the entries and keeps each
# one's leading facts, so evidence that can occupy any entry survives at the same prompt size.
# The cap decides HOW MANY bytes reach the summarizer and this decides WHICH ones.
# `per_entry_head` ships as the default: the adoption study measured it losing no paired case on
# any workload, recovering every middle-critical case `head_tail` could not finish, and spending no
# extra summary bytes, and the policy-change audit replays every published cell bit-identically
# under both trims in either direction -- so a number measured under `head_tail` stands unrestated.
# `head_tail` remains selectable for a run that wants to reproduce the retired behavior.
SUMMARY_TRIM_HEAD_TAIL = "head_tail"
SUMMARY_TRIM_PER_ENTRY_HEAD = "per_entry_head"
SUMMARY_TRIM_STRATEGIES: tuple[str, ...] = (SUMMARY_TRIM_HEAD_TAIL, SUMMARY_TRIM_PER_ENTRY_HEAD)
DEFAULT_SUMMARY_TRIM_STRATEGY = SUMMARY_TRIM_PER_ENTRY_HEAD

TRIMMING_POLICIES = frozenset({POLICY_OBSERVATION_CAP, POLICY_COMPACT})


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """One context-management policy plus the constants it is parameterized by."""

    name: str = POLICY_FULL
    observation_cap_chars: int = DEFAULT_OBSERVATION_CAP_CHARS
    observation_head_share: float = OBSERVATION_HEAD_SHARE
    keep_last_n: int = DEFAULT_KEEP_LAST_N
    compact_share: float = DEFAULT_COMPACT_SHARE
    compact_keep_recent: int = DEFAULT_COMPACT_KEEP_RECENT
    summary_input_cap: str = DEFAULT_SUMMARY_INPUT_CAP
    summary_trim_strategy: str = DEFAULT_SUMMARY_TRIM_STRATEGY

    def __post_init__(self) -> None:
        if self.name not in CONTEXT_POLICIES:
            raise ValueError(
                f"unknown context policy: {self.name!r}; choose from {CONTEXT_POLICIES}"
            )
        if not 0.0 < self.observation_head_share < 1.0:
            raise ValueError(
                f"observation_head_share must be in (0, 1), got {self.observation_head_share}"
            )
        if self.summary_input_cap not in SUMMARY_INPUT_CAPS:
            raise ValueError(
                f"unknown summary input cap: {self.summary_input_cap!r}; "
                f"choose from {SUMMARY_INPUT_CAPS}"
            )
        if self.summary_trim_strategy not in SUMMARY_TRIM_STRATEGIES:
            raise ValueError(
                f"unknown summary trim strategy: {self.summary_trim_strategy!r}; "
                f"choose from {SUMMARY_TRIM_STRATEGIES}"
            )
