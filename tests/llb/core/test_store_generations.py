"""Immutable store-generation resolution and publishing."""

import os

from llb.core.store_generations import (
    generation_timestamp,
    new_generation_paths,
    publish_generation,
    resolve_store_dir,
)

META = "store_meta.json"


def _write_meta(directory, mtime=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / META
    path.write_text("{}", encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_resolves_base_when_no_generations(tmp_path):
    assert resolve_store_dir(tmp_path, META) == tmp_path  # no store at all -> base unchanged
    _write_meta(tmp_path)
    assert resolve_store_dir(tmp_path, META) == tmp_path


def test_newest_generation_wins_and_rollback_restores_base(tmp_path):
    _write_meta(tmp_path, mtime=1000)
    gen1 = tmp_path / "generations" / "20990101T000000Z"
    gen2 = tmp_path / "generations" / "20990102T000000Z"
    _write_meta(gen1, mtime=2000)
    _write_meta(gen2, mtime=3000)
    assert resolve_store_dir(tmp_path, META) == gen2
    (gen2 / META).unlink()
    gen2.rmdir()  # rollback: delete the newest generation
    assert resolve_store_dir(tmp_path, META) == gen1


def test_a_newer_base_rebuild_takes_over(tmp_path):
    gen = tmp_path / "generations" / "20990101T000000Z"
    _write_meta(gen, mtime=2000)
    _write_meta(tmp_path, mtime=3000)  # build-index rewrote the base afterwards
    assert resolve_store_dir(tmp_path, META) == tmp_path


def test_tie_prefers_the_generation(tmp_path):
    _write_meta(tmp_path, mtime=2000)
    gen = tmp_path / "generations" / "20990101T000000Z"
    _write_meta(gen, mtime=2000)
    assert resolve_store_dir(tmp_path, META) == gen


def test_staging_dirs_are_ignored_until_published(tmp_path):
    _write_meta(tmp_path, mtime=1000)
    staging, final = new_generation_paths(tmp_path, "20990101T000000Z")
    _write_meta(staging, mtime=2000)
    assert resolve_store_dir(tmp_path, META) == tmp_path  # hidden staging never resolves
    publish_generation(staging, final)
    assert resolve_store_dir(tmp_path, META) == final


def test_generation_names_never_collide(tmp_path):
    first_staging, first_final = new_generation_paths(tmp_path, "20990101T000000Z")
    first_staging.mkdir(parents=True)
    publish_generation(first_staging, first_final)
    _second_staging, second_final = new_generation_paths(tmp_path, "20990101T000000Z")
    assert second_final.name == "20990101T000000Z-2"


def test_generation_timestamp_is_utc_second_resolution():
    stamp = generation_timestamp()
    assert len(stamp) == 16 and stamp.endswith("Z") and "T" in stamp
