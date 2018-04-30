PANDOC=pandoc
LATEX2PDF=pdflatex
BIBER=biber

pdf:
	$(PANDOC) paper.md -s -o paper.tex --filter=pandoc-crossref --filter=pandoc-citeproc --biblatex
	$(LATEX2PDF) paper.tex
	$(BIBER) paper.bcf
	$(LATEX2PDF) paper.tex
	

.PHONY: pdf

