"""Shared corpus fixtures for the ontology-drafting tests.

Not collected by pytest (module name does not start with `test_`). The two toy documents are
shared across the split test modules (`test_ontology_draft`, `test_ontology_extract`,
`test_ontology_coverage`); `test_ontology_draft` re-exports them (and `fake_endpoint`) for
`test_ontology_resume`.
"""

DOC1 = "# Київ\n\nКиїв є столицею України. Місто розташоване на річці Дніпро.\n"
DOC2 = "# Львів\n\nЛьвів є культурним центром заходу. Місто засноване у 1256 році.\n"
