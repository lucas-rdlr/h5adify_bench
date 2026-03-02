#!/usr/bin/env bash
set -euo pipefail

# Spatial Hubs Figure Generator - LaTeX/Bash Version
# Letters overlaid on images WITHOUT boxes

OUTBASE="${1:-panel_part1}"

# Find required tools
TECTONIC="$(command -v tectonic || true)"
PDFLATEX="$(command -v pdflatex || true)"
PDFTOPPM="$(command -v pdftoppm || true)"
GS="$(command -v gs || true)"

if [[ -z "${TECTONIC}" && -z "${PDFLATEX}" ]]; then
  echo "[ERROR] Need tectonic or pdflatex in PATH."
  echo "Install: sudo apt-get install texlive-latex-base"
  echo "    or: conda install -c conda-forge tectonic"
  exit 1
fi

echo "===================================================================="
echo "Creating LaTeX source..."
echo "===================================================================="

# Generate LaTeX document
cat > "${OUTBASE}.tex" <<'EOF'
\documentclass[border=5mm]{standalone}
\usepackage{silence}
\WarningsOff*
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{multirow}
\setlength{\parindent}{0pt}

% Safe include for paths with underscores
\newcommand{\img}[2][]{\includegraphics[#1]{\detokenize{#2}}}

% Panel macro: letter overlaid on image (NO BOX)
% #1 = letter
% #2 = title
% #3 = image path
% #4 = includegraphics options
\newcommand{\Panel}[4]{%
  \begin{minipage}[t]{\linewidth}
    \centering
    % Title above
    {\itshape #2\par}
    \vspace{2mm}
    % Image with letter overlay
    \begin{tikzpicture}
      \node[anchor=north west, inner sep=0] (img) {\img[#4]{#3}};
      % Letter overlaid at top-left (NO BOX, just text)
      \node[
        anchor=north west,
        inner sep=0,
        font=\bfseries\Large,
        xshift=2mm,
        yshift=5mm
      ] at (img.north west) {#1};
    \end{tikzpicture}
  \end{minipage}%
}

\begin{document}

\begin{minipage}{200mm}
\centering

\begin{tabular}{cc}

% Row 1
\begin{minipage}[t]{0.48\linewidth}
\Panel{A}{}{/home/user/Documents/h5adify_bench/h5adify_bench/figures/part1/Fig1_accuracy_per_study.png}{width=\linewidth}
\end{minipage}
&
\multirow{3}{*}{
\begin{minipage}[t]{0.48\linewidth}
\Panel{D}{}{/home/user/Documents/h5adify_bench/h5adify_bench/figures/part1/Fig9_spider_overall_and_per_dataset.png}{width=\linewidth}
\end{minipage}
}
\\
% Row 2
\begin{minipage}[t]{0.48\linewidth}
\Panel{B}{}{/home/user/Documents/h5adify_bench/h5adify_bench/figures/part1/Fig4_runtime_per_study.png}{width=\linewidth}
\end{minipage}
\\
% Row 3
\begin{minipage}[t]{0.48\linewidth}
\Panel{C}{}{/home/user/Documents/h5adify_bench/h5adify_bench/figures/part1/Fig5_field_coverage.png}{width=\linewidth}
\end{minipage}

\end{tabular}

% Row 1: A B C
\begin{minipage}[t]{0.48\linewidth}
\Panel{E}{}
{/home/user/Documents/h5adify_bench/h5adify_bench/figures/part1/Fig2_field_heatmap_quantitative_v2.png}
{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\linewidth}
\Panel{F}{}
{/home/user/Documents/h5adify_bench/h5adify_bench/figures/part1/Fig8_sankey_like_FIXED.png}
{width=\linewidth}
\end{minipage}
\\
\begin{minipage}[t]{0.48\linewidth}
\Panel{G}{}
{/home/user/Documents/h5adify_bench/h5adify_bench/figures/part1/columns_barplot.png}
{width=\linewidth}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\linewidth}
\Panel{H}{}
{/home/user/Documents/h5adify_bench/h5adify_bench/figures/part1/columns_heatmap_acc.png}
{width=\linewidth}
\end{minipage}

\end{minipage}
\end{document}
EOF

echo "✓ LaTeX source created: ${OUTBASE}.tex"
echo ""
echo "===================================================================="
echo "Compiling PDF..."
echo "===================================================================="

# Compile
if [[ -n "${TECTONIC}" ]]; then
  echo "Using tectonic..."
  "${TECTONIC}" -c minimal "${OUTBASE}.tex"
else
  echo "Using pdflatex..."
  "${PDFLATEX}" -interaction=nonstopmode -halt-on-error "${OUTBASE}.tex" >/dev/null 2>&1
fi

echo "✓ PDF created: ${OUTBASE}.pdf"
echo ""
echo "===================================================================="
echo "Converting to PNG and JPG..."
echo "===================================================================="

# Rasterize
if [[ -n "${PDFTOPPM}" ]]; then
  echo "Using pdftoppm (300 DPI)..."
  pdftoppm -png -r 300 -singlefile "${OUTBASE}.pdf" "${OUTBASE}"
  pdftoppm -jpeg -r 300 -singlefile "${OUTBASE}.pdf" "${OUTBASE}"
  echo "✓ PNG: ${OUTBASE}.png"
  echo "✓ JPG: ${OUTBASE}.jpg"
elif [[ -n "${GS}" ]]; then
  echo "Using ghostscript (300 DPI)..."
  gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r300 \
     -sOutputFile="${OUTBASE}.png" "${OUTBASE}.pdf" 2>/dev/null
  gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=jpeg -r300 -dJPEGQ=95 \
     -sOutputFile="${OUTBASE}.jpg" "${OUTBASE}.pdf" 2>/dev/null
  echo "✓ PNG: ${OUTBASE}.png"
  echo "✓ JPG: ${OUTBASE}.jpg"
else
  echo "⚠ No pdftoppm or gs found - only PDF created"
  echo "Install: sudo apt-get install poppler-utils"
fi

echo ""
echo "===================================================================="
echo "✓ SUCCESS! Figure created with letters on images (no boxes)"
echo "===================================================================="
echo "Output files:"
echo "  ${OUTBASE}.pdf"
[[ -f "${OUTBASE}.png" ]] && echo "  ${OUTBASE}.png"
[[ -f "${OUTBASE}.jpg" ]] && echo "  ${OUTBASE}.jpg"
echo "===================================================================="

echo "===================================================================="
echo "Cleaning up auxiliary files..."
echo "===================================================================="

# Delete auxiliary files
rm -f "${OUTBASE}.aux" "${OUTBASE}.log" "${OUTBASE}.tex" "${OUTBASE}.jpg"

echo "✓ Cleanup complete. Remaining files:"
echo "  ${OUTBASE}.pdf"
[[ -f "${OUTBASE}.png" ]] && echo "  ${OUTBASE}.png"
