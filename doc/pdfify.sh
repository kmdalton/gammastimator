#!/bin/bash

for i in `ls fig`;do
    inkscape --export-area-drawing --export-pdf=pdf/${i%.svg}.pdf fig/$i
done
