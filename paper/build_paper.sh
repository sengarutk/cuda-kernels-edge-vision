#!/usr/bin/env bash
# Builds paper/main.pdf and an arXiv/Overleaf source archive (paper/build/arxiv.zip).
# Run `make report figures` first so paper/generated/ and paper/figures/ match results/.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "pdflatex not found; install a TeX distribution (e.g. texlive-latex-extra)." >&2
    exit 1
fi

mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex >/dev/null
(cd build && BIBINPUTS=.. bibtex main >/dev/null)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex >/dev/null
if grep -E "undefined|Citation .* undefined|\?\?" build/main.log | grep -v "^Package rerunfilecheck" ; then
    echo "LaTeX reported undefined references or result keys (see above)." >&2
    exit 1
fi
cp build/main.pdf main.pdf
echo "[built] paper/main.pdf"

rm -f build/arxiv.zip
zip -q -r build/arxiv.zip main.tex references.bib generated/results.tex figures/*.pdf build/main.bbl
echo "[built] paper/build/arxiv.zip"
