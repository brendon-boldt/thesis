main_file = src/main.tex

# .PHONY: output/document.pdf clean assets/linguistic-dag.pdf
.PHONY: clean

SOURCE_FILES = $(shell find -type f -regex '.*\.\(tex\|bib\)')
ASSETS = $(shell find assets/ -type f)

ifeq ($(shell command -v dot),)
DOT = : Graphviz not found, skipping: dot
else
DOT = dot
endif

output/document.pdf: $(SOURCE_FILES) $(ASSETS)
	latexmk \
			-f \
			-g \
			-pdf \
			-interaction=nonstopmode \
			-outdir=output/ \
			-jobname=document \
			$(main_file)

assets/linguistic-dag.pdf: src/figures/linguistic-dag.dot
	$(DOT) src/figures/linguistic-dag.dot -Tpdf -o assets/linguistic-dag.pdf


clean:
	rm -rf output/
