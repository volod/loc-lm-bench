"""Scanned-book recovery: the resumable page-and-stage run behind `llb book-ocr`.

`runner` starts and resumes a run, `stage_exec` decides per page and stage whether to skip, reuse,
or execute, `cache` owns keys and atomic commits, `ledger` projects the stage records into the page
manifest, and `stages` is the seam every recovery step registers behind.
"""
