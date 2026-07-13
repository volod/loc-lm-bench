# shellcheck shell=bash
# PDF draft corpus staging and draft-stat estimation.

pdf_draft_doc_ids() {
  if [ "$QS_PDF_DRAFT_DOCS" = "all" ]; then
    find "$QS_PDF_MD" -maxdepth 1 -type f -name '*.md' -printf '%f\n' \
      | sed 's/\.md$//' \
      | sort
  else
    printf '%s\n' $QS_PDF_DRAFT_DOCS
  fi
}

stage_pdf_draft_corpus() {
  if [ "$QS_PDF_DRAFT_MD" = "$QS_PDF_MD" ]; then
    echo "ERROR: QUICKSTART_PDF_DRAFT_MD must differ from QUICKSTART_PDF_MD" >&2
    exit 2
  fi
  rm -rf "$QS_PDF_DRAFT_MD"
  mkdir -p "$QS_PDF_DRAFT_MD"
  local doc n=0
  while IFS= read -r doc; do
    test -n "$doc" || continue
    test -f "$QS_PDF_MD/$doc.md" || { echo "ERROR: missing $QS_PDF_MD/$doc.md" >&2; exit 1; }
    test -f "$QS_PDF_MD/$doc.citations.json" || {
      echo "ERROR: missing $QS_PDF_MD/$doc.citations.json" >&2
      exit 1
    }
    cp -R "$QS_PDF_MD/$doc.md" "$QS_PDF_MD/$doc.citations.json" "$QS_PDF_DRAFT_MD/"
    n=$((n + 1))
    printf '[draft-corpus] staged %s\n' "$doc"
  done < <(pdf_draft_doc_ids)
  if [ "$n" -eq 0 ]; then
    echo "ERROR: no markdown documents found under $(rel_path "$QS_PDF_MD")" >&2
    exit 1
  fi
  result "draft input corpus: $(rel_path "$QS_PDF_DRAFT_MD") ($n docs)"
}

pdf_draft_stats() {
  local docs chars chunk windows calls tok_s
  docs="$(find "$QS_PDF_DRAFT_MD" -maxdepth 1 -type f -name '*.md' | wc -l | tr -d ' ')"
  # wc -m (characters), not -c (bytes): Cyrillic UTF-8 is ~2 bytes/char, and the extractor
  # windows by CHARACTERS, so byte counts would overestimate the workload ~2x.
  chars="$(find "$QS_PDF_DRAFT_MD" -maxdepth 1 -type f -name '*.md' -print0 \
    | xargs -0 wc -m 2>/dev/null \
    | awk 'END {print $1 + 0}')"
  chunk="${QS_DRAFT_EXTRACT_MAX_CHARS:-12000}"
  windows=$(( (chars + chunk - 1) / chunk ))
  calls=$(( windows + QS_DRAFT_MAX_ITEMS ))
  tok_s=0
  if [ -f "$(pdf_bench_json)" ] && [ "$QS_DRAFT_ENDPOINT" = "local" ]; then
    tok_s="$(quickstart_py speed "$(pdf_bench_json)" "$QS_DRAFT_MODEL")"
  fi
  if [ "${tok_s%.*}" = "0" ]; then
    case "$QS_DRAFT_MODEL" in
      *12B*|*12b*) tok_s=24 ;;
      *27B*|*27b*|*31b*|*35b*) tok_s=8 ;;
      *24b*|*26b*) tok_s=12 ;;
      *) tok_s=16 ;;
    esac
  fi
  local seconds hours
  seconds="$(awk -v calls="$calls" -v tok="$tok_s" 'BEGIN {printf "%.0f", calls * 500 / tok * 2}')"
  hours="$(awk -v sec="$seconds" 'BEGIN {printf "%.1f", sec / 3600}')"
  printf '%s docs, %s chars, about %s extraction windows + %s draft calls, %s hours' \
    "$docs" "$chars" "$windows" "$QS_DRAFT_MAX_ITEMS" "$hours"
}

