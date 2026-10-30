#!/bin/bash
# Renders every .dot diagram of this folder to PDF (slides), SVG and PNG. Needs Graphviz (apt install graphviz).
cd "$(dirname "$0")"
for f in *.dot; do
    name="${f%.dot}"
    dot -Tpdf "$f" -o "$name.pdf" && dot -Tsvg "$f" -o "$name.svg" && dot -Tpng -Gdpi=150 "$f" -o "$name.png" \
        && echo "rendered $name"
done
