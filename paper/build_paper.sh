#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ -f "/home/sengar/dev/.venv/bin/activate" ]; then
    source /home/sengar/dev/.venv/bin/activate
fi

echo "=== 1. Verifying Evidence Integrity ==="
python3 ../scripts/verify_evidence_integrity.py

echo "=== 2. Generating Publication Vector Figures ==="
python3 generate_figures.py

if command -v pdflatex &> /dev/null; then
    echo "=== 3. Compiling Research Paper PDF ==="
    pdflatex -interaction=nonstopmode main.tex
    bibtex main || true
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
    echo "[✓] Research paper successfully compiled: paper/main.pdf"
else
    echo "[!] pdflatex not found on host. Vector figures, LaTeX source, and references written cleanly."
fi

echo "=== 4. Packaging Clean Overleaf / arXiv Archive ==="
python3 -c "
import zipfile, os
for zip_path in ['paper_camera_ready.zip', 'paper.zip']:
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in ['main.tex', 'references.bib', 'build_paper.sh', 'generate_figures.py']:
            if os.path.exists(f):
                z.write(f, f)
        if os.path.exists('figures'):
            for fig in sorted(os.listdir('figures')):
                if fig.endswith('.pdf'):
                    z.write(os.path.join('figures', fig), os.path.join('figures', fig))
    print(f'[✓] Successfully packaged: paper/{zip_path}')
"
