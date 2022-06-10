VENV           = venv
SYSTEM_PYTHON  = $(or $(shell which python3), $(shell which python))
PYTHON         = $(or $(wildcard venv/bin/python), $(SYSTEM_PYTHON))

.PHONY: docs

venv-build: venv-create
	$(PYTHON) -m pip install -r src/python/requirements.txt
	$(PYTHON) -m pip install -e src/python/.

venv-create:
	rm -rf $(VENV)
	$(SYSTEM_PYTHON) -m virtualenv $(VENV)

venv-up:
	bash -c "source $(VENV)/bin/activate && jupyter lab"

test:
	$(PYTHON) -m pytest -s -vv --pyargs vimure

docs: 
	Rscript -e 'vimure::install_vimure(version="src/python/");setwd("src/R");pkgdown::build_site()'
	git add docs && git commit -m "Update docs"
	git push origin `git subtree split --prefix docs develop`:gh-pages --force
	rm -rf docs/