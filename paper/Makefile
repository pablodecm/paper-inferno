.DEFAUT_GOAL := pdf_draft
PANDOC=pandoc
LATEX2PDF=pdflatex
BIBER=biber
BIBTEX=bibtex

pdf_draft: diagram appendices
	$(PANDOC) draft.yaml paper.md -s -o paper.tex --include-after-body=appendices.tex \
	--filter=pandoc-crossref --filter=pandoc-citeproc --biblatex --wrap=preserve
	$(LATEX2PDF) paper.tex
	$(BIBER) paper.bcf
	$(LATEX2PDF) paper.tex

appendices: appendices.md
	$(PANDOC) appendices.md -o appendices.tex --filter=pandoc-crossref \
	--filter=pandoc-citeproc --biblatex --wrap=preserve

diagram: diagram.tex
	$(LATEX2PDF) diagram.tex
	mv diagram.pdf gfx/figure1.pdf

arxiv_tar: diagram appendices
	$(PANDOC) arxiv.yaml paper.md -s -o paper.tex --include-after-body=appendices.tex \
	--filter=pandoc-crossref --filter=pandoc-citeproc --biblatex
	mkdir -p arxiv
	cp -r paper.tex arxiv/.
	cp -r table.tex arxiv/.
	cp -r table_syst.tex arxiv/.
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
	--natbib --wrap=preserve
	cp -r table.tex cpc_inferno/.
	cp -r table_syst.tex cpc_inferno/.
	cp -r gfx cpc_inferno/.
	cp -r bibliography.bib cpc_inferno/.
	tar --exclude='./cpc_inferno/*.latex' -czvf cpc_inferno.tar.gz cpc_inferno

cpc_diff: cpc_tar
	git show cpc_diff:paper/paper.md > paper_old.md
	git show cpc_diff:paper/appendices.md > appendices_old.md
	$(PANDOC) appendices_old.md -o appendices_old.tex --filter=pandoc-crossref \
	--filter=pandoc-citeproc --biblatex --wrap=preserve
	$(PANDOC) paper_old.md -s -o cpc_inferno/paper_old.tex \
	--template cpc_inferno/cpc_article.latex \
	--include-after-body=appendices_old.tex \
	--filter=pandoc-crossref --filter=pandoc-citeproc \
	--natbib --wrap=preserve
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
