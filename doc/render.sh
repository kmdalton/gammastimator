#!/bin/bash

./clean.sh

mkdir renders

for i in fig/*.svg;do
    outfn=${i#fig/}
    outfn=${outfn%.svg}.pdf
    inkscape --export-area-drawing --export-pdf=renders/$outfn $i
done


for i in fig/*.py;do
    python $i
done


pdflatex main.tex
bibtex main.aux
pdflatex main.tex
