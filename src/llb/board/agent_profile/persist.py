"""Write one composed profile bundle under `$DATA_DIR/agent-profile/<run_timestamp>/`.

No new evidence root: the bundle holds only the composition and its rationale, and every value in
it points back into the per-lane root that measured it.
"""

import json
from pathlib import Path

from llb.board.agent_profile.model import METHOD, PROFILE_JSON, PROFILE_MD, AgentProfile
from llb.board.agent_profile.render import format_profile_md, profile_payload


def write_profile(profile: AgentProfile, data_dir: Path | str) -> dict[str, Path]:
    """Persist `agent_profile.json` + `profile.md`; returns both paths."""
    from llb.bench.common import new_run_timestamp

    _, run_timestamp = new_run_timestamp()
    out_dir = Path(data_dir) / METHOD / run_timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / PROFILE_JSON
    md_path = out_dir / PROFILE_MD
    json_path.write_text(
        json.dumps(profile_payload(profile), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(format_profile_md(profile) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
