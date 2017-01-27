

jupyter nbconvert --template _templates/tutorial_rst ../notebooks/Example\ -\ filling\ missing\ bands.ipynb --to rst --output-dir _static
mv _static/Example\ -\ filling\ missing\ bands.rst .


jupyter nbconvert --template _templates/tutorial_rst ../notebooks/Tutorial\ -\ getting\ started\ with\ Delight.ipynb --to rst --output-dir _static
mv _static/Tutorial\ -\ getting\ started\ with\ Delight.rst .
