.DEFAUT_GOAL := pdf_draft
PANDOC=pandoc
LATEX2PDF=pdflatex
BIBER=biber
BIBTEX=bibtex

pdf_draft: diagram appendices
	$(PANDOC) draft.yaml paper.md -s -o paper.tex --include-after-body=appendices.tex --filter=pandoc-crossref --filter=pandoc-citeproc --biblatex
	$(LATEX2PDF) paper.tex
	$(BIBER) paper.bcf
	$(LATEX2PDF) paper.tex

appendices: appendices.md
	$(PANDOC) appendices.md -o appendices.tex --filter=pandoc-crossref --filter=pandoc-citeproc --biblatex

diagram: diagram.tex
	$(LATEX2PDF) diagram.tex
	mv diagram.pdf gfx/figure1.pdf

arxiv_tar: diagram appendices
	$(PANDOC) arxiv.yaml paper.md -s -o paper.tex --include-after-body=appendices.tex --filter=pandoc-crossref --filter=pandoc-citeproc --biblatex
	mkdir -p arxiv
	cp -r paper.tex arxiv/.
	cp -r table.tex arxiv/.
	cp -r gfx arxiv/.
	cp -r bibliography.bib arxiv/.
	cp nips_2018.sty arxiv/.
	$(LATEX2PDF) paper.tex
	$(BIBER) paper.bcf
	mv paper.aux arxiv/.
	mv paper.bbl arxiv/.
	tar -czvf arxiv.tar.gz arxiv

arxiv_pdf: arxiv_tar
	cd arxiv; $(LATEX2PDF) paper.tex

cpc_tar: diagram appendices
	$(PANDOC) paper.md -s -o cpc_inferno/paper.tex \
	--template cpc_inferno/cpc_article.latex \
	--include-after-body=appendices.tex \
	--filter=pandoc-crossref --filter=pandoc-citeproc \
	--natbib
	cp -r table.tex cpc_inferno/.
	cp -r gfx cpc_inferno/.
	cp -r bibliography.bib cpc_inferno/.
	tar --exclude='./cpc_inferno/*.latex' -czvf cpc_inferno.tar.gz cpc_inferno

cpc_diff: cpc_tar
	git show v2.1.0:paper/paper.md > paper_old.md
	$(PANDOC) paper_old.md -s -o cpc_inferno/paper_old.tex \
	--template cpc_inferno/cpc_article.latex \
	--include-after-body=appendices.tex \
	--filter=pandoc-crossref --filter=pandoc-citeproc \
	--natbib
	latexdiff cpc_inferno/paper_old.tex cpc_inferno/paper.tex > cpc_inferno/paper_diff.tex
	cd cpc_inferno; $(LATEX2PDF) paper_diff.tex
	cd cpc_inferno; $(BIBTEX) paper_diff
	cd cpc_inferno; $(LATEX2PDF) paper_diff.tex; $(LATEX2PDF) paper_diff.tex


cpc_pdf: cpc_tar
	cd cpc_inferno; $(LATEX2PDF) paper.tex
	cd cpc_inferno; $(BIBTEX) paper
	cd cpc_inferno; $(LATEX2PDF) paper.tex; $(LATEX2PDF) paper.tex

cpc_clean:
	cd cpc_inferno; rm -f *.pdf *.out *.aux *.bbl *.blg *.log *.toc \
	                      *.ptb *.tod *.fls *.fdb_latexmk *.lof *.spl
