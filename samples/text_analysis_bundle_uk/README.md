# Committed Ukrainian text-analysis fixture

This repo-authored synthetic fixture contains exactly two short Ukrainian Markdown documents.
`provenance.json` records their hashes and classifies the content as a synthetic repo fixture. It
is the pinned input for `make frontier-ua-draft-probe`; the command still requires interactive
egress consent and an explicit provider spend cap.

The labels and corpus remain useful for local text-analysis smoke tests. The frontier probe uses
only `corpus/`; it does not send the committed labels to the provider.
