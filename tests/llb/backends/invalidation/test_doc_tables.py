"""Reading model identities out of the delivered docs' markdown tables, and what is NOT read."""

from pathlib import Path

from llb.backends.invalidation.doc_tables import read_doc_tree, read_model_cells

TABLE = """# Throughput

| model | served artifact | tok/s |
| --- | --- | ---: |
| `qwen3.8-27b` (Qwen 3.8, CURRENT) | `qwen3.8:27b` | 10.38 |
| `gemma-4-12b-it-w4a16` | `gemma4:12b` | 29.84 |

Prose naming `qwen3.8:27b` outside any table.

| finding | note |
| --- | --- |
| `qwen3.6:27b` | a table with no model column |
"""


def _write(root: Path, name: str, body: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_a_model_column_table_yields_one_cell_per_declared_column(tmp_path) -> None:
    cells = read_model_cells(_write(tmp_path, "telemetry.md", TABLE))

    assert [cell.identity for cell in cells] == [
        "qwen3.8-27b",
        "qwen3.8:27b",
        "gemma-4-12b-it-w4a16",
        "gemma4:12b",
    ]


def test_each_cell_carries_the_row_and_the_header_an_operator_would_open(tmp_path) -> None:
    cells = read_model_cells(_write(tmp_path, "telemetry.md", TABLE))

    assert [cell.line for cell in cells] == [5, 5, 6, 6]
    assert {cell.header_line for cell in cells} == {3}
    assert {cell.column for cell in cells} == {"model", "served artifact"}


def test_a_table_whose_header_declares_no_model_column_is_not_read(tmp_path) -> None:
    """A markdown table has no schema, so the header is the only statement of what a row is."""
    cells = read_model_cells(_write(tmp_path, "telemetry.md", TABLE))

    assert "qwen3.6:27b" not in {cell.identity for cell in cells}


def test_an_unticked_cell_is_a_label_rather_than_an_identity(tmp_path) -> None:
    body = "| model | note |\n| --- | --- |\n| qwen3.8-27b | not backticked |\n"

    assert read_model_cells(_write(tmp_path, "labels.md", body)) == []


def test_the_first_non_row_line_ends_the_table(tmp_path) -> None:
    """A row separated from the table by prose belongs to whatever comes next, not to this one."""
    body = TABLE + "\n| `lapa-v0.1.2-instruct` | detached | 1 |\n"
    identities = {cell.identity for cell in read_model_cells(_write(tmp_path, "t.md", body))}

    assert "lapa-v0.1.2-instruct" not in identities


def test_the_tree_walk_reads_every_markdown_file_under_it(tmp_path) -> None:
    _write(tmp_path, "a.md", TABLE)
    _write(tmp_path, "nested/b.md", TABLE)
    _write(tmp_path, "nested/c.txt", TABLE)

    assert len(read_doc_tree(tmp_path)) == 8
