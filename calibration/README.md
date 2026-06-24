# Calibration worksheets (M3.8)

Judge-calibration worksheets -- the CSVs the human rates so `make calibration-score` can compute
the Spearman rho that gates the LLM judge. They carry irreducibly-human ratings, so they are kept
in **two roots**:

| Root | Tracked? | For |
| --- | --- | --- |
| `calibration/` (this dir) | yes -- committed | **permanent** sets: survive a clone, reusable |
| `$DATA_DIR/llb/calibration/` | no -- gitignored | **temporary/generated** sets (in progress) |

This separation avoids a brittle `.gitignore` exception: this whole directory is committed, and
generated worksheets live elsewhere -- there is no per-file glob that could ignore the wrong thing.

## Naming + auto-routing (`CAL_NAME`)

The worksheet path is `$(CAL_DIR)/$(CAL_NAME).csv` (`CAL_WS`). `CAL_NAME` labels the calibration
target, and the worksheet **auto-routes** by name:

- names in `CAL_PERMANENT` (Makefile) -> `calibration/<name>.csv` (here, committed)
- any other name -> `$DATA_DIR/llb/calibration/<name>.csv` (temporary, gitignored)

| Use case | command | worksheet |
| --- | --- | --- |
| Committed canonical goldset (default) | `make calibration-run` | `calibration/ua_squad_postedited_v1.csv` |
| Goldset from the skeleton | `make calibration-run CAL_NAME=skeleton …` | `$DATA_DIR/llb/calibration/skeleton.csv` |
| Goldset from a text corpus | `make calibration-run CAL_NAME=<corpus> …` | `$DATA_DIR/llb/calibration/<corpus>.csv` |

### Persisting a generated set

```
cp "$DATA_DIR/llb/calibration/<name>.csv" calibration/        # copy into the tracked dir
# add <name> to CAL_PERMANENT in the Makefile, then it routes here automatically:
make calibration-score CAL_NAME=<name>
git add calibration/<name>.csv
```

## Columns

`item_id, split, provenance, question, reference_answer, model_answer, human_answer,
human_rating, human_note, human_status, judge_rating`. The human owns `human_*`; `model_answer`
and `judge_rating` come from `calibration-run`. See the
[calibration-tooling guide](../docs/guides/calibration-tooling.md).

## Reusing on a fresh clone

The committed worksheet already carries the human + judge ratings, so you can score directly
without re-running the judge:

```
make calibration-score                 # RATINGS defaults to calibration/ua_squad_postedited_v1.csv
```

To re-rate or extend it, `make calibration-rate`; to refresh `judge_rating` (e.g. a new judge),
`make calibration-run` (it merges your existing human ratings by `item_id`).
