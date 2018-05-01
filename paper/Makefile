.DEFAUT_GOAL := pdf
PANDOC=pandoc
LATEX2PDF=pdflatex
BIBER=biber

pdf: diagram.pdf paper.tex
	$(PANDOC) paper.md -s -o paper.tex --filter=pandoc-crossref --filter=pandoc-citeproc --biblatex
	$(LATEX2PDF) paper.tex
	$(BIBER) paper.bcf
	$(LATEX2PDF) paper.tex

diagram.pdf: diagram.tex
	$(LATEX2PDF) diagram.tex

