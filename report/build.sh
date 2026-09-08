#!/usr/bin/env bash
# Render ../churn_case/SUMMARY.md to report.pdf. SUMMARY.md stays the single
# source of truth; nothing here duplicates its content.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$HERE/../SUMMARY.md"

command -v pandoc >/dev/null || { echo "pandoc is required (brew install pandoc)"; exit 1; }
command -v pdflatex >/dev/null || { echo "a TeX distribution providing pdflatex is required"; exit 1; }
[ -f "$SOURCE" ] || { echo "missing source: $SOURCE"; exit 1; }

pandoc "$SOURCE" \
  --from=markdown+tex_math_single_backslash \
  --metadata-file="$HERE/metadata.yaml" \
  --resource-path="$HERE/.." \
  --pdf-engine=pdflatex \
  --lua-filter="$HERE/columns.lua" \
  -V documentclass=article \
  -V fontsize=11pt \
  -H "$HERE/report.tex" \
  -o "$HERE/report.pdf"

echo "wrote report.pdf"
