# Quickstart orchestration targets.
##@ Quickstart Workflows

.PHONY: \
	quickstart-goldset quickstart-goldset-setup quickstart-goldset-rag \
	quickstart-goldset-models quickstart-goldset-eval quickstart-goldset-security \
	quickstart-goldset-prompt quickstart-pdf-corpus quickstart-pdf-corpus-convert \
	quickstart-pdf-corpus-index quickstart-pdf-corpus-draft quickstart-pdf-corpus-graph \
	quickstart-pdf-corpus-validate quickstart-pdf-corpus-review quickstart-pdf-corpus-accept \
	quickstart-pdf-corpus-score quickstart-corpus quickstart-corpus-convert \
	quickstart-corpus-index quickstart-corpus-draft quickstart-corpus-graph \
	quickstart-corpus-validate

quickstart-goldset: ## Quickstart all-in-one: committed goldset -> RAG -> model prep -> sweep -> backend matrix -> security -> prompts
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" goldset

quickstart-goldset-setup: ## Quickstart group: venv, CUDA tier detection, serving config generation
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" goldset-setup

quickstart-goldset-rag: ## Quickstart group: build and validate committed-goldset RAG index
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" goldset-rag

quickstart-goldset-models: ## Quickstart group: list and prepare model candidates
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" goldset-models

quickstart-goldset-eval: ## Quickstart group: model-family sweep and backend platform matrix
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" goldset-eval

quickstart-goldset-security: ## Quickstart group: run model security tests as a separate benchmark tier
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" goldset-security

quickstart-goldset-prompt: ## Quickstart group: prompt candidates; set QUICKSTART_PROMPT_ID=<id> to pin and score
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" goldset-prompt

quickstart-pdf-corpus: ## Quickstart all-in-one: PDF corpus -> RAG -> full goldset/ontology draft -> graph -> validation
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus

quickstart-pdf-corpus-convert: ## Quickstart group: convert QUICKSTART_PDF_SOURCE PDFs to markdown/citations
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-convert

quickstart-pdf-corpus-index: ## Quickstart group: build full PDF-corpus RAG index
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-index

quickstart-pdf-corpus-draft: ## Quickstart group: select drafter and draft full unverified PDF goldset/ontology
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-draft

quickstart-pdf-corpus-graph: ## Quickstart group: build graph artifacts from the draft bundle
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-graph

quickstart-pdf-corpus-validate: ## Quickstart group: validate draft structure and retrieval metrics
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-validate

quickstart-pdf-corpus-review: ## Quickstart human gate: review verify_sample.csv interactively
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-review

quickstart-pdf-corpus-accept: ## Quickstart human gate: emit accepted ledger after review
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-accept

quickstart-pdf-corpus-score: ## Quickstart continuation: score accepted PDF corpus/goldset
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" pdf-corpus-score

quickstart-corpus: ## Quickstart all-in-one: mixed txt/md/pdf corpus -> RAG -> full goldset/ontology draft -> graph -> validation (QUICKSTART_CORPUS_SRC=)
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" corpus

quickstart-corpus-convert: ## Quickstart group: ingest QUICKSTART_CORPUS_SRC (txt/md/pdf) into one corpus
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" corpus-convert

quickstart-corpus-index: ## Quickstart group: build full mixed-corpus RAG index
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" corpus-index

quickstart-corpus-draft: ## Quickstart group: select drafter and draft full unverified goldset/ontology (QUICKSTART_CORPUS_RESUME=<bundle> resumes)
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" corpus-draft

quickstart-corpus-graph: ## Quickstart group: build graph artifacts from the mixed-corpus draft bundle
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" corpus-graph

quickstart-corpus-validate: ## Quickstart group: validate mixed-corpus draft structure and retrieval
	@bash "$(PROJECT_ROOT)/scripts/quickstart.sh" corpus-validate
