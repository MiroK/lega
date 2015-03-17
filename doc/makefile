all: lega_theory.pdf view

lega_theory.pdf: lega_theory.tex lega_theory.bib
	pdflatex lega_theory.tex
	bibtex lega_theory.aux
	pdflatex lega_theory.tex
	pdflatex lega_theory.tex

view:
	open lega_theory.pdf
