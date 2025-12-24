PYTHON=python3

.PHONY: phase1 phase2 phase3 plot-surface repair heston clean-reports

# Phase 1: normalize + sanity
phase1:
	./scripts/phase1.sh

# Phase 2: detection (bid-ask aware)
phase2:
	$(PYTHON) scripts/phase2_detect.py --dataset both --sample-size 200 --strike-bucket 1

# Phase 3: smoothing (calls-only)
phase3:
	$(PYTHON) scripts/phase3_smooth.py --dataset both --grid-size 60 --max-plots 6 --rate 0 --dividend 0

# Plot 3D IV surface (requires phase3 outputs). Override GRID/PARAMS/QUOTE/OUT as needed.
plot-surface:
	$(PYTHON) scripts/plot_iv_surface.py \
		--grid $(or $(GRID),reports/phase3/1545/grid_fit.csv) \
		--params $(or $(PARAMS),reports/phase3/1545/fit_params.csv) \
		$(if $(QUOTE),--quote-date $(QUOTE),) \
		$(if $(TITLE),--title $(TITLE),) \
		$(if $(OUT),--output $(OUT),) \
		--show

# Placeholders for upcoming phases
repair:
	@echo "Repair step not implemented yet (Phase 4 placeholder)."

heston:
	@echo "Heston calibration not implemented yet (Phase 5 placeholder)."

# Clean generated reports (optional)
clean-reports:
	rm -rf reports/phase2 reports/phase3

