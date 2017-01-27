#!/usr/bin/bash

for f in "Example - filling missing bands"  "Tutorial - getting started with Delight"
do
	echo "Processing $f"
    jupyter nbconvert --template _templates/tutorial_rst ../notebooks/"$f".ipynb --to rst --output-dir _static
    mv _static/"$f".rst .
done
