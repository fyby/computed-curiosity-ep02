# Reproduce everything:  make all      (about 1.5-2 h on a 10-core Apple M4 Mac mini)
PY ?= python

all:
	$(PY) run.py all

validate:
	$(PY) run.py validate

production:
	$(PY) run.py e1 e2 e3 replays

results:
	$(PY) run.py results previews

smoke:
	MILLION_SMOKE=1 MILLION_RESULTS=$(CURDIR)/results_smoke $(PY) run.py all

.PHONY: all validate production results smoke
