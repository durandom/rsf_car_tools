.PHONY: all clean clusters predict

# Create output directory
out:
	mkdir -p out

# Run clustering analysis
clusters: out
	python -m powersteering.cli assets/rsf --select-sample 20 --html out/clusters.html --tui=false

# Run prediction analysis
predict: out
	python -m powersteering.cli assets/rsf --generate --html out/prediction.html --tui=false

# Generate statistical plots
plots: out
	python -m powersteering.cli assets/rsf --stats weight,drivetrain,steering --html out/plots.html --tui=false

# Default target
all: clusters predict plots

# Clean generated files
clean:
	rm -rf out/

# Get version from pyproject.toml
version:
	@python -c "import tomli; print(tomli.load(open('pyproject.toml', 'rb'))['project']['version'])"

# Create a new release
release:
	$(eval VERSION := $(shell make version))
	@echo "Creating release v$(VERSION)"
	git tag "v$(VERSION)"
	git push origin "v$(VERSION)"
