"""Prospective robotics benchmark constants shared by validation and scoring."""

METHOD_NAME = "robotics-rag"
SCHEMA_VERSION = 1
LANE_NO_RETRIEVAL = "no_retrieval"
LANE_RETRIEVAL = "retrieval"
LANE_REFERENCE = "reference"
MANDATORY_FAULT_CLASSES = frozenset(
    {
        "stale_state",
        "wrong_device",
        "limit",
        "approval",
        "injection",
        "emergency_stop",
        "concurrency",
        "ambiguous_retry",
    }
)
DECISION_PROPOSE = "propose"
DECISION_REFUSE = "refuse"
DECISION_ESCALATE = "escalate"
EXPECTED_COMPLETE = "complete"
EXPECTED_REFUSE = "refuse"
