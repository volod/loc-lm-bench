# shellcheck shell=bash
# Usage text and target dispatch.
# Sourced by scripts/quickstart.sh, which loads the siblings below first; this fragment never
# sources them itself, so it declares them for the llb_* call check.
# llb-requires: track_a.sh
# llb-requires: track_b.sh
# llb-requires: track_c.sh

llb_quickstart_usage() {
  cat <<'EOF'
Usage: scripts/quickstart.sh <target>

Targets:
  goldset                  committed-goldset setup + RAG + model prep + sweep + backend matrix + security + prompts
  goldset-setup            venv, GPU tier detection, serving config generation
  goldset-rag              build and validate committed-goldset RAG index
  goldset-models           list and prepare model candidates
  goldset-eval             sweep model candidates and run backend platform matrix
  goldset-security         run model security tests as a separate benchmark tier
  goldset-prompt           prepare prompt candidates; pin/eval when QUICKSTART_PROMPT_ID is set
  pdf-corpus               PDF corpus conversion + index + draft + graph + validation
  pdf-corpus-convert       PDF to markdown conversion
  pdf-corpus-index         build full PDF-corpus RAG index
  pdf-corpus-draft         select drafter, prepare full draft corpus, and draft unverified goldset
  pdf-corpus-graph         build graph artifacts from the draft bundle
  pdf-corpus-validate      validate draft structure and retrieval
  pdf-corpus-review        interactive human review of verify_sample.csv
  pdf-corpus-accept        emit accepted ledger after review
  pdf-corpus-score         run accepted corpus/goldset through goldset scoring
  corpus                   mixed txt/md/pdf ingest + index + draft + graph + validation
  corpus-convert           ingest a mixed txt/md/pdf corpus into one .md/.txt corpus
  corpus-index             build full mixed-corpus RAG index
  corpus-draft             select drafter and draft unverified goldset (QUICKSTART_CORPUS_RESUME resumes)
  corpus-graph             build graph artifacts from the mixed-corpus draft bundle
  corpus-validate          validate mixed-corpus draft structure and retrieval
EOF
}

llb_run_target() {
  local target="$1"
  case "$target" in
    goldset) llb_track_a_all ;;
    goldset-setup) llb_track_a_setup ;;
    goldset-rag) llb_track_a_rag ;;
    goldset-models) llb_track_a_models ;;
    goldset-eval) llb_track_a_eval ;;
    goldset-security) llb_track_a_security ;;
    goldset-prompt) llb_track_a_prompt ;;
    pdf-corpus) llb_track_b_all ;;
    pdf-corpus-convert) llb_track_b_convert ;;
    pdf-corpus-index) llb_track_b_index ;;
    pdf-corpus-draft) llb_track_b_draft ;;
    pdf-corpus-graph) llb_track_b_graph ;;
    pdf-corpus-validate) llb_track_b_validate ;;
    pdf-corpus-review) llb_track_b_review ;;
    pdf-corpus-accept) llb_track_b_accept ;;
    pdf-corpus-score) llb_track_b_after_accept ;;
    corpus) llb_track_c_all ;;
    corpus-convert) llb_track_c_convert ;;
    corpus-index) llb_use_corpus_paths; llb_track_b_index ;;
    corpus-draft) llb_use_corpus_paths; llb_track_c_draft ;;
    corpus-graph) llb_use_corpus_paths; llb_track_b_graph ;;
    corpus-validate) llb_use_corpus_paths; llb_track_b_validate ;;
    help|-h|--help|"") llb_quickstart_usage ;;
    *) echo "ERROR: unknown quickstart target: $target" >&2; llb_quickstart_usage >&2; exit 2 ;;
  esac
}
