"""What `pyvenv.cfg` recorded vs what the interpreter is now (llb.build.venv_interpreter).

This is the fact `make venv` was missing: uv replaces an environment whose recorded version has
moved, so the recorded version is what decides between a reuse and a full reinstall. The restamp
half is bounded by the ABI -- a patch move keeps every compiled wheel in the venv loadable, a minor
move does not -- and these fix that boundary.
"""

import sys
from pathlib import Path

from llb.build import venv_interpreter
from tests.llb.build._venv_fixtures import (
    OTHER_MINOR,
    PATCHED_AWAY,
    RUNNING_VERSION,
    write_venv,
)


def test_version_triple_reads_both_writers_of_pyvenv_cfg():
    # uv writes three components, CPython's own venv module writes five.
    assert venv_interpreter.version_triple("3.13.14") == (3, 13, 14)
    assert venv_interpreter.version_triple("3.13.14.final.0") == (3, 13, 14)
    # Anything that is not a release triple is not comparable, and guessing would be worse.
    assert venv_interpreter.version_triple("3.13") is None
    assert venv_interpreter.version_triple("") is None


def test_read_config_parses_the_keys_uv_writes(tmp_path):
    config = venv_interpreter.read_config(write_venv(tmp_path / ".venv"))

    assert config[venv_interpreter.VERSION_KEY] == RUNNING_VERSION
    assert config[venv_interpreter.HOME_KEY] == str(Path(sys.executable).parent)


def test_read_config_of_a_missing_venv_is_empty(tmp_path):
    assert venv_interpreter.read_config(tmp_path / "nowhere") == {}


def test_base_executable_wins_over_home(tmp_path):
    """CPython's own `venv` records the exact executable; uv records only its directory."""
    venv_dir = write_venv(tmp_path / ".venv", home=str(tmp_path / "gone"))
    path = venv_dir / venv_interpreter.PYVENV_CFG
    path.write_text(
        f"{path.read_text(encoding='utf-8')}"
        f"{venv_interpreter.BASE_EXECUTABLE_KEY} = {sys.executable}\n",
        encoding="utf-8",
    )

    resolved = venv_interpreter.base_interpreter(venv_interpreter.read_config(venv_dir), venv_dir)

    assert resolved == Path(sys.executable)


def test_a_removed_interpreter_resolves_to_nothing(tmp_path):
    venv_dir = write_venv(tmp_path / ".venv", home=str(tmp_path / "removed-python"))

    config = venv_interpreter.read_config(venv_dir)

    assert venv_interpreter.base_interpreter(config, venv_dir) is None


def _read(venv_dir: Path) -> venv_interpreter.Interpreter:
    return venv_interpreter.read_interpreter(venv_interpreter.read_config(venv_dir), venv_dir)


def test_a_patch_move_is_restampable_and_a_minor_move_is_not(tmp_path):
    patched = _read(write_venv(tmp_path / "patch", version_info=PATCHED_AWAY))
    minor = _read(write_venv(tmp_path / "minor", version_info=OTHER_MINOR))
    current = _read(write_venv(tmp_path / "current"))

    assert patched.moved and patched.patch_move
    assert minor.moved and not minor.patch_move
    assert not current.moved and not current.patch_move


def test_restamp_records_the_running_version_and_keeps_every_other_key(tmp_path):
    venv_dir = write_venv(tmp_path / ".venv", version_info=PATCHED_AWAY)

    assert venv_interpreter.restamp(venv_dir) == venv_interpreter.RESTAMP_OK

    config = venv_interpreter.read_config(venv_dir)
    assert config[venv_interpreter.VERSION_KEY] == RUNNING_VERSION
    # uv reads the rest of this file too, so a rewrite that loses a key is its own bug.
    assert config["implementation"] == "CPython"
    assert config["include-system-site-packages"] == "false"
    assert config[venv_interpreter.HOME_KEY] == str(Path(sys.executable).parent)


def test_restamp_of_a_current_venv_changes_nothing(tmp_path):
    venv_dir = write_venv(tmp_path / ".venv")
    before = (venv_dir / venv_interpreter.PYVENV_CFG).read_text(encoding="utf-8")

    assert venv_interpreter.restamp(venv_dir) == venv_interpreter.RESTAMP_OK
    assert (venv_dir / venv_interpreter.PYVENV_CFG).read_text(encoding="utf-8") == before


def test_restamp_refuses_a_minor_move_because_the_abi_did_not_hold(tmp_path, caplog):
    """The one case a restamp would be a lie: new stdlib, new layout, unloadable extensions."""
    venv_dir = write_venv(tmp_path / ".venv", version_info=OTHER_MINOR)

    with caplog.at_level("ERROR"):
        status = venv_interpreter.restamp(venv_dir)

    assert status == venv_interpreter.RESTAMP_REFUSED
    assert "MINOR move" in caplog.text and "RECREATE_VENV=1" in caplog.text
    assert venv_interpreter.read_config(venv_dir)[venv_interpreter.VERSION_KEY] == OTHER_MINOR


def test_restamp_reports_a_venv_it_cannot_read(tmp_path, caplog):
    with caplog.at_level("ERROR"):
        assert venv_interpreter.restamp(tmp_path / "nowhere") == venv_interpreter.RESTAMP_UNKNOWN

    assert venv_interpreter.PYVENV_CFG in caplog.text
