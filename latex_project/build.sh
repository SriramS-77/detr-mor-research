#!/usr/bin/env bash
# Build main.pdf or title_page.pdf with Tectonic (https://tectonic-typesetting.github.io).
# Usage: bash build.sh [main|title_page]        (default: main)
# Set TECTONIC=/path/to/tectonic if it is not on PATH.
set -e
cd "$(dirname "$0")"
TEC="${TECTONIC:-tectonic}"
doc="${1:-main}"
python analysis/make_tables.py >/dev/null
mkdir -p build
cp "$doc.tex" sn-jnl.cls sn-basic.bst build/
if [ -f references.bib ]; then cp references.bib build/; fi
rm -rf build/generated && cp -r generated build/
cd build
"$TEC" -X compile --keep-logs "$doc.tex"
cp "$doc.pdf" ../
