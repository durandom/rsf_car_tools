.PHONY: all clean clusters predict

# Create output directory
out:
	mkdir -p out

# Run clustering analysis
clusters: out
	python -m powersteering.cli assets/rsf --select-sample 20 --html out/clusters.html

# Run prediction analysis
predict: out
	python -m powersteering.cli assets/rsf --generate --html out/prediction.html

# Default target
all: clusters predict

# Clean generated files
clean:
	rm -rf out/
